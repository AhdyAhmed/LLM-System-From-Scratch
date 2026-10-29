package com.scratchllm.platform.identity;

import com.scratchllm.platform.identity.IdentityDtos.LoginRequest;
import com.scratchllm.platform.identity.IdentityDtos.RegisterRequest;
import com.scratchllm.platform.identity.IdentityDtos.UserProfileResponse;
import com.scratchllm.platform.infrastructure.RedisRateLimiterService;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import java.util.Locale;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseCookie;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

/**
 * Day011 & Day012: Registration, Redis-backed login with session fixation rotation, logout, and /api/v1/me.
 */
@RestController
public class AuthController {

    private final IdentityService identityService;
    private final RedisSessionStore sessionStore;
    private final RedisRateLimiterService rateLimiterService;

    public AuthController(
        IdentityService identityService,
        RedisSessionStore sessionStore,
        RedisRateLimiterService rateLimiterService
    ) {
        this.identityService = identityService;
        this.sessionStore = sessionStore;
        this.rateLimiterService = rateLimiterService;
    }

    @PostMapping("/auth/register")
    public ResponseEntity<UserProfileResponse> register(
        @Valid @RequestBody RegisterRequest request,
        HttpServletRequest servletRequest
    ) {
        String rateKey = servletRequest.getRemoteAddr() + ":register";
        rateLimiterService.enforceAuthLimit(rateKey);
        UserProfileResponse profile = identityService.registerUser(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(profile);
    }

    @PostMapping("/auth/login")
    public ResponseEntity<UserProfileResponse> login(
        @Valid @RequestBody LoginRequest request,
        HttpServletRequest servletRequest
    ) {
        String normalizedEmail = request.email().trim().toLowerCase(Locale.ROOT);
        rateLimiterService.enforceAuthLimit(normalizedEmail);

        UserProfileResponse profile = identityService.authenticate(request);
        String existingSessionId = extractSessionCookie(servletRequest);
        String newSessionId = sessionStore.rotateAndCreateSession(
            existingSessionId,
            profile.user_id(),
            profile.email()
        );
        ResponseCookie sessionCookie = sessionStore.buildSessionCookie(newSessionId);

        return ResponseEntity.ok()
            .header(HttpHeaders.SET_COOKIE, sessionCookie.toString())
            .body(profile);
    }

    @PostMapping("/auth/logout")
    public ResponseEntity<Void> logout(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        HttpServletRequest servletRequest
    ) {
        String sessionId = principal != null ? principal.sessionId() : extractSessionCookie(servletRequest);
        sessionStore.invalidateSession(sessionId);
        ResponseCookie expiredCookie = sessionStore.buildExpiredSessionCookie();

        return ResponseEntity.noContent()
            .header(HttpHeaders.SET_COOKIE, expiredCookie.toString())
            .build();
    }

    @GetMapping("/api/v1/me")
    public ResponseEntity<UserProfileResponse> me(
        @AuthenticationPrincipal AuthenticatedPrincipal principal
    ) {
        return ResponseEntity.ok(identityService.getUserProfile(principal.userId()));
    }

    public static String extractSessionCookie(HttpServletRequest request) {
        Cookie[] cookies = request.getCookies();
        if (cookies == null) {
            return null;
        }
        for (Cookie cookie : cookies) {
            if (RedisSessionStore.SESSION_COOKIE_NAME.equals(cookie.getName())) {
                return cookie.getValue();
            }
        }
        return null;
    }
}
