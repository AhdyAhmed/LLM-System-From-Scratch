package com.scratchllm.platform.identity;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.Cookie;
import java.util.Map;
import java.util.UUID;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

/**
 * Day011 & Day012 integration tests against live PostgreSQL 16 and Redis 7.4:
 * - Account creation, BCrypt hashing, duplicate-account handling, and uniform auth errors
 * - Redis-backed login/logout, session fixation rotation, CSRF enforcement, and cookie flags
 */
@SpringBootTest
@AutoConfigureMockMvc
class Phase2AuthAndRedisSessionIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Test
    void registersUserWithBcryptHashRejectsDuplicatesAndEnforcesCsrf() throws Exception {
        CsrfBundle csrf = bootstrapCsrf();
        String email = "alice-" + UUID.randomUUID() + "@example.com";

        // Missing CSRF header must be rejected with 403 CSRF_INVALID
        mockMvc.perform(post("/auth/register")
                .cookie(csrf.xsrfCookie())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of(
                    "email", email,
                    "password", "StrongPass#2026!",
                    "workspace_name", "Alice Lab"
                ))))
            .andExpect(status().isForbidden())
            .andExpect(jsonPath("$.error_code").value("CSRF_INVALID"));

        // Valid registration creates user, default workspace, and OWNER membership
        MvcResult regResult = mockMvc.perform(post("/auth/register")
                .cookie(csrf.xsrfCookie())
                .header("X-CSRF-TOKEN", csrf.token())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of(
                    "email", email,
                    "password", "StrongPass#2026!",
                    "workspace_name", "Alice Lab"
                ))))
            .andExpect(status().isCreated())
            .andExpect(jsonPath("$.email").value(email))
            .andExpect(jsonPath("$.workspaces[0].name").value("Alice Lab"))
            .andExpect(jsonPath("$.workspaces[0].role").value("OWNER"))
            .andReturn();

        JsonNode profile = objectMapper.readTree(regResult.getResponse().getContentAsString());
        UUID userId = UUID.fromString(profile.get("user_id").asText());

        // Verify stored password in PostgreSQL is a BCrypt hash ($2a$12$...) and never plaintext
        String storedHash = jdbcTemplate.queryForObject(
            "SELECT password_hash FROM users WHERE id = ?",
            String.class,
            userId
        );
        assertThat(storedHash)
            .startsWith("$2a$12$")
            .doesNotContain("StrongPass#2026!");

        // Duplicate registration returns 409 ACCOUNT_ALREADY_EXISTS
        mockMvc.perform(post("/auth/register")
                .cookie(csrf.xsrfCookie())
                .header("X-CSRF-TOKEN", csrf.token())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of(
                    "email", email.toUpperCase(),
                    "password", "AnotherStrongPass#2026!",
                    "workspace_name", "Duplicate Lab"
                ))))
            .andExpect(status().isConflict())
            .andExpect(jsonPath("$.error_code").value("ACCOUNT_ALREADY_EXISTS"));
    }

    @Test
    void rotatesRedisSessionOnLoginEnforcesCookieFlagsAndInvalidatesOnLogout() throws Exception {
        CsrfBundle csrf = bootstrapCsrf();
        String email = "bob-" + UUID.randomUUID() + "@example.com";

        mockMvc.perform(post("/auth/register")
                .cookie(csrf.xsrfCookie())
                .header("X-CSRF-TOKEN", csrf.token())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of(
                    "email", email,
                    "password", "BobSecurePass#2026!",
                    "workspace_name", "Bob Workspace"
                ))))
            .andExpect(status().isCreated());

        // First login with a pre-existing attacker session cookie rotates session ID (session fixation defense)
        Cookie preAuthSession = new Cookie("SESSION", "attacker-fixed-session-id");
        MvcResult login1 = mockMvc.perform(post("/auth/login")
                .cookie(csrf.xsrfCookie(), preAuthSession)
                .header("X-CSRF-TOKEN", csrf.token())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of(
                    "email", email,
                    "password", "BobSecurePass#2026!"
                ))))
            .andExpect(status().isOk())
            .andExpect(header().exists(HttpHeaders.SET_COOKIE))
            .andReturn();

        String setCookieHeader = login1.getResponse().getHeader(HttpHeaders.SET_COOKIE);
        assertThat(setCookieHeader)
            .contains("SESSION=")
            .contains("HttpOnly")
            .contains("SameSite=Lax")
            .doesNotContain("attacker-fixed-session-id");

        Cookie sessionCookie = login1.getResponse().getCookie("SESSION");
        assertThat(sessionCookie).isNotNull();

        // Authenticated /api/v1/me succeeds using the Redis-backed SESSION cookie
        mockMvc.perform(get("/api/v1/me").cookie(sessionCookie))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.email").value(email));

        // Logout invalidates the session in Redis and expires the cookie
        mockMvc.perform(post("/auth/logout")
                .cookie(csrf.xsrfCookie(), sessionCookie)
                .header("X-CSRF-TOKEN", csrf.token()))
            .andExpect(status().isNoContent());

        // Subsequent request with the logged-out SESSION cookie is rejected with 401 UNAUTHORIZED
        mockMvc.perform(get("/api/v1/me").cookie(sessionCookie))
            .andExpect(status().isUnauthorized())
            .andExpect(jsonPath("$.error_code").value("UNAUTHORIZED"));
    }

    private CsrfBundle bootstrapCsrf() throws Exception {
        MvcResult res = mockMvc.perform(get("/auth/csrf"))
            .andExpect(status().isOk())
            .andReturn();
        JsonNode json = objectMapper.readTree(res.getResponse().getContentAsString());
        Cookie xsrfCookie = res.getResponse().getCookie("XSRF-TOKEN");
        return new CsrfBundle(json.get("token").asText(), xsrfCookie);
    }

    private record CsrfBundle(String token, Cookie xsrfCookie) {}
}
