package com.scratchllm.platform.platform;

import java.util.Map;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.csrf.CookieCsrfTokenRepository;
import org.springframework.security.web.csrf.CsrfToken;
import org.springframework.security.web.csrf.CsrfTokenRequestAttributeHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Day005 / Day007 platform security foundation:
 * Exposes GET /auth/csrf for headless cookie-jar clients and enforces CSRF on state-changing requests.
 */
@Configuration
@EnableWebSecurity
public class SecurityFoundationConfig {

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder(12);
    }

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        CsrfTokenRequestAttributeHandler requestHandler = new CsrfTokenRequestAttributeHandler();
        requestHandler.setCsrfRequestAttributeName("_csrf");

        CookieCsrfTokenRepository csrfRepository = new CookieCsrfTokenRepository();
        csrfRepository.setHeaderName("X-CSRF-TOKEN");

        http
            .csrf(csrf -> csrf
                .csrfTokenRepository(csrfRepository)
                .csrfTokenRequestHandler(requestHandler)
            )
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/auth/csrf", "/actuator/health", "/actuator/info").permitAll()
                .anyRequest().authenticated()
            );

        return http.build();
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
