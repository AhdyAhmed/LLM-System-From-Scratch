package com.scratchllm.platform.infrastructure;

import com.scratchllm.platform.platform.ApiException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DataAccessException;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.script.DefaultRedisScript;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

/**
 * Day017: Atomic Redis Lua rate limiter with bounded TTLs and distinct auth/user key namespaces.
 * Fails closed (503 REDIS_UNAVAILABLE) if Redis is unreachable.
 */
@Service
public class RedisRateLimiterService {

    /**
     * Single atomic Lua operation:
     * Increments key counter, sets EXPIRE on first hit (or heals missing TTL),
     * and returns {allowed (1 or 0), current_count, ttl_seconds}.
     */
    private static final String ATOMIC_RATE_LIMIT_LUA = """
        local current = redis.call('INCR', KEYS[1])
        local window = tonumber(ARGV[1])
        local limit = tonumber(ARGV[2])
        if current == 1 then
            redis.call('EXPIRE', KEYS[1], window)
        else
            local ttl = redis.call('TTL', KEYS[1])
            if ttl < 0 then
                redis.call('EXPIRE', KEYS[1], window)
            end
        end
        local ttl_now = redis.call('TTL', KEYS[1])
        if ttl_now < 1 then
            ttl_now = window
        end
        if current > limit then
            return {0, current, ttl_now}
        end
        return {1, current, ttl_now}
        """;

    @SuppressWarnings("rawtypes")
    private final DefaultRedisScript<List> rateLimitScript;
    private final StringRedisTemplate redisTemplate;
    private final int authMaxRequests;
    private final int authWindowSeconds;
    private final int runMaxRequests;
    private final int runWindowSeconds;

    public RedisRateLimiterService(
        StringRedisTemplate redisTemplate,
        @Value("${scratchllm.ratelimit.auth-max-requests:20}") int authMaxRequests,
        @Value("${scratchllm.ratelimit.auth-window-seconds:60}") int authWindowSeconds,
        @Value("${scratchllm.ratelimit.run-max-requests:15}") int runMaxRequests,
        @Value("${scratchllm.ratelimit.run-window-seconds:60}") int runWindowSeconds
    ) {
        this.redisTemplate = redisTemplate;
        this.authMaxRequests = authMaxRequests;
        this.authWindowSeconds = authWindowSeconds;
        this.runMaxRequests = runMaxRequests;
        this.runWindowSeconds = runWindowSeconds;
        this.rateLimitScript = new DefaultRedisScript<>(ATOMIC_RATE_LIMIT_LUA, List.class);
    }

    public RateLimitResult checkAuthLimit(String actorKey) {
        return evaluateLimit("scratchllm:ratelimit:auth:" + actorKey, authWindowSeconds, authMaxRequests);
    }

    public RateLimitResult checkRunSubmissionLimit(String userId) {
        return evaluateLimit("scratchllm:ratelimit:run:" + userId, runWindowSeconds, runMaxRequests);
    }

    public RateLimitResult evaluateCustomLimit(String keyNamespace, int windowSeconds, int maxRequests) {
        return evaluateLimit("scratchllm:ratelimit:" + keyNamespace, windowSeconds, maxRequests);
    }

    public void enforceAuthLimit(String actorKey) {
        RateLimitResult res = checkAuthLimit(actorKey);
        if (!res.allowed()) {
            throw new ApiException(
                HttpStatus.TOO_MANY_REQUESTS,
                "RATE_LIMITED",
                "Too many authentication requests; retry after window expires.",
                Map.of("retry_after_seconds", res.ttlSeconds(), "scope", "auth")
            );
        }
    }

    public void enforceRunSubmissionLimit(String userId) {
        RateLimitResult res = checkRunSubmissionLimit(userId);
        if (!res.allowed()) {
            throw new ApiException(
                HttpStatus.TOO_MANY_REQUESTS,
                "RATE_LIMITED",
                "Run submission rate limit exceeded; retry after window expires.",
                Map.of("retry_after_seconds", res.ttlSeconds(), "scope", "run_submission")
            );
        }
    }

    @SuppressWarnings("unchecked")
    private RateLimitResult evaluateLimit(String redisKey, int windowSeconds, int maxRequests) {
        try {
            List<Long> raw = (List<Long>) redisTemplate.execute(
                rateLimitScript,
                Collections.singletonList(redisKey),
                String.valueOf(windowSeconds),
                String.valueOf(maxRequests)
            );
            if (raw == null || raw.size() < 3) {
                throw new ApiException(
                    HttpStatus.SERVICE_UNAVAILABLE,
                    "REDIS_UNAVAILABLE",
                    "Rate limiter returned empty response from Redis."
                );
            }
            boolean allowed = raw.get(0) != null && raw.get(0) == 1L;
            long count = raw.get(1) != null ? raw.get(1) : 0L;
            long ttl = raw.get(2) != null ? raw.get(2) : windowSeconds;
            return new RateLimitResult(allowed, count, ttl, maxRequests);
        } catch (DataAccessException ex) {
            throw new ApiException(
                HttpStatus.SERVICE_UNAVAILABLE,
                "REDIS_UNAVAILABLE",
                "Redis rate-limit store is unavailable; rejecting new admission."
            );
        }
    }

    public record RateLimitResult(
        boolean allowed,
        long currentCount,
        long ttlSeconds,
        int limit
    ) {}
}
