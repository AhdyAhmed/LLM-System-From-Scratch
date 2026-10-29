package com.scratchllm.platform.platform;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.scratchllm.platform.identity.AuthController;
import com.scratchllm.platform.identity.AuthenticatedPrincipal;
import com.scratchllm.platform.identity.RedisSessionStore;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.security.web.csrf.CookieCsrfTokenRepository;
import org.springframework.security.web.csrf.CsrfException;
import org.springframework.security.web.csrf.CsrfToken;
import org.springframework.security.web.csrf.CsrfTokenRequestAttributeHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * Day005 / Day011 / Day012 platform security configuration:
 * - Exposes GET /auth/csrf for cookie-jar clients
 * - Enforces X-CSRF-TOKEN on all state-changing requests
 * - Resolves shared Redis sessions via RedisSessionStore
 * - Returns structured ApiErrorResponse JSON on 401/403/503
 */
@Configuration
@EnableWebSecurity
public class SecurityFoundationConfig {

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder(12);
    }

    @Bean
    public SecurityFilterChain securityFilterChain(
        HttpSecurity http,
        RedisSessionStore sessionStore,
        ObjectMapper objectMapper
    ) throws Exception {
        CsrfTokenRequestAttributeHandler requestHandler = new CsrfTokenRequestAttributeHandler();
        requestHandler.setCsrfRequestAttributeName("_csrf");

        CookieCsrfTokenRepository csrfRepository = new CookieCsrfTokenRepository();
        csrfRepository.setHeaderName("X-CSRF-TOKEN");

        OncePerRequestFilter redisSessionFilter = new OncePerRequestFilter() {
            @Override
            protected void doFilterInternal(
                HttpServletRequest request,
                HttpServletResponse response,
                FilterChain filterChain
            ) throws ServletException, IOException {
                String sessionId = AuthController.extractSessionCookie(request);
                if (sessionId != null && !sessionId.isBlank()) {
                    try {
                        Optional<AuthenticatedPrincipal> resolved = sessionStore.resolveSession(sessionId);
                        resolved.ifPresent(principal -> {
                            UsernamePasswordAuthenticationToken auth = new UsernamePasswordAuthenticationToken(
                                principal,
                                null,
                                List.of(new SimpleGrantedAuthority("ROLE_USER"))
                            );
                            SecurityContextHolder.getContext().setAuthentication(auth);
                        });
                    } catch (ApiException ex) {
                        writeJsonError(response, request, objectMapper, ex.getStatus(), ex.getErrorCode(), ex.getMessage());
                        return;
                    }
                }
                filterChain.doFilter(request, response);
            }
        };

        http
            .sessionManagement(sm -> sm.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .csrf(csrf -> csrf
                .csrfTokenRepository(csrfRepository)
                .csrfTokenRequestHandler(requestHandler)
            )
            .addFilterBefore(redisSessionFilter, UsernamePasswordAuthenticationFilter.class)
            .exceptionHandling(eh -> eh
                .authenticationEntryPoint((request, response, authException) ->
                    writeJsonError(
                        response,
                        request,
                        objectMapper,
                        HttpStatus.UNAUTHORIZED,
                        "UNAUTHORIZED",
                        "Authentication session is missing, expired, or invalid."
                    )
                )
                .accessDeniedHandler((request, response, accessDeniedException) -> {
                    String code = (accessDeniedException instanceof CsrfException) ? "CSRF_INVALID" : "FORBIDDEN";
                    String msg = (accessDeniedException instanceof CsrfException)
                        ? "Missing or invalid X-CSRF-TOKEN header or CSRF cookie."
                        : "Access to the requested resource is forbidden.";
                    writeJsonError(response, request, objectMapper, HttpStatus.FORBIDDEN, code, msg);
                })
            )
            .authorizeHttpRequests(auth -> auth
                .dispatcherTypeMatchers(
                    jakarta.servlet.DispatcherType.ASYNC,
                    jakarta.servlet.DispatcherType.ERROR
                ).permitAll()
                .requestMatchers(
                    "/auth/csrf",
                    "/auth/register",
                    "/auth/login",
                    "/actuator/health",
                    "/actuator/info"
                ).permitAll()
                .anyRequest().authenticated()
            );

        return http.build();
    }

    private static void writeJsonError(
        HttpServletResponse response,
        HttpServletRequest request,
        ObjectMapper objectMapper,
        HttpStatus status,
        String errorCode,
        String message
    ) throws IOException {
        response.setStatus(status.value());
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        ApiErrorResponse payload = new ApiErrorResponse(
            errorCode,
            message,
            RequestIdFilter.resolveRequestId(request),
            OffsetDateTime.now(ZoneOffset.UTC),
            null
        );
        objectMapper.writeValue(response.getWriter(), payload);
    }

    @RestController
    public static class CsrfBootstrapController {

        @GetMapping(value = "/auth/csrf", produces = MediaType.APPLICATION_JSON_VALUE)
        public Map<String, String> getCsrfToken(CsrfToken csrfToken) {
            return Map.of(
                "header_name", csrfToken.getHeaderName(),
                "parameter_name", csrfToken.getParameterName(),
                "token", csrfToken.getToken()
            );
        }
    }
}
