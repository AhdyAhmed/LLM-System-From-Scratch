package com.scratchllm.platform.identity;

import com.scratchllm.platform.platform.ApiException;
import java.security.SecureRandom;
import java.time.Duration;
import java.util.Base64;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DataAccessException;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseCookie;
import org.springframework.stereotype.Service;

/**
 * Day012: Shared Redis session store with session fixation rotation, bounded TTL,
 * environment-specific cookie flags, and fail-closed Redis unavailability handling.
 */
@Service
public class RedisSessionStore {

    public static final String SESSION_COOKIE_NAME = "SESSION";
    private static final String KEY_PREFIX = "scratchllm:session:";
    private final SecureRandom secureRandom = new SecureRandom();
    private final StringRedisTemplate redisTemplate;
    private final long sessionTtlSeconds;
    private final boolean cookieSecure;

    public RedisSessionStore(
        StringRedisTemplate redisTemplate,
        @Value("${scratchllm.auth.session-ttl-seconds:28800}") long sessionTtlSeconds,
        @Value("${scratchllm.auth.cookie-secure:false}") boolean cookieSecure
    ) {
        this.redisTemplate = redisTemplate;
        this.sessionTtlSeconds = sessionTtlSeconds;
        this.cookieSecure = cookieSecure;
    }

    /**
     * Rotates any existing session ID (preventing session fixation) and stores the new session in Redis.
     */
    public String rotateAndCreateSession(String existingSessionId, UUID userId, String email) {
        try {
            if (existingSessionId != null && !existingSessionId.isBlank()) {
                redisTemplate.delete(KEY_PREFIX + existingSessionId);
            }
            byte[] randomBytes = new byte[32];
            secureRandom.nextBytes(randomBytes);
            String newSessionId = Base64.getUrlEncoder().withoutPadding().encodeToString(randomBytes);
            String key = KEY_PREFIX + newSessionId;

            redisTemplate.opsForHash().putAll(key, Map.of(
                "user_id", userId.toString(),
                "email", email
            ));
            redisTemplate.expire(key, Duration.ofSeconds(sessionTtlSeconds));
            return newSessionId;
        } catch (DataAccessException ex) {
            throw new ApiException(
                HttpStatus.SERVICE_UNAVAILABLE,
                "REDIS_UNAVAILABLE",
                "Session store is unavailable; rejecting authentication."
            );
        }
    }

    public Optional<AuthenticatedPrincipal> resolveSession(String sessionId) {
        if (sessionId == null || sessionId.isBlank()) {
            return Optional.empty();
        }
        try {
            String key = KEY_PREFIX + sessionId;
            Map<Object, Object> entries = redisTemplate.opsForHash().entries(key);
            if (entries == null || entries.isEmpty()) {
                return Optional.empty();
            }
            Object rawUserId = entries.get("user_id");
            Object rawEmail = entries.get("email");
            if (rawUserId == null || rawEmail == null) {
                return Optional.empty();
            }
            return Optional.of(new AuthenticatedPrincipal(
                UUID.fromString(rawUserId.toString()),
                rawEmail.toString(),
                sessionId
            ));
        } catch (DataAccessException ex) {
            throw new ApiException(
                HttpStatus.SERVICE_UNAVAILABLE,
                "REDIS_UNAVAILABLE",
                "Session store is unavailable; failing closed."
            );
        }
    }

    public void invalidateSession(String sessionId) {
        if (sessionId == null || sessionId.isBlank()) {
            return;
        }
        try {
            redisTemplate.delete(KEY_PREFIX + sessionId);
        } catch (DataAccessException ex) {
            throw new ApiException(
                HttpStatus.SERVICE_UNAVAILABLE,
                "REDIS_UNAVAILABLE",
                "Session store is unavailable during logout."
            );
        }
    }

    public ResponseCookie buildSessionCookie(String sessionId) {
        return ResponseCookie.from(SESSION_COOKIE_NAME, sessionId)
            .httpOnly(true)
            .secure(cookieSecure)
            .sameSite("Lax")
            .path("/")
            .maxAge(Duration.ofSeconds(sessionTtlSeconds))
            .build();
    }

    public ResponseCookie buildExpiredSessionCookie() {
        return ResponseCookie.from(SESSION_COOKIE_NAME, "")
            .httpOnly(true)
            .secure(cookieSecure)
            .sameSite("Lax")
            .path("/")
            .maxAge(Duration.ZERO)
            .build();
    }
}
