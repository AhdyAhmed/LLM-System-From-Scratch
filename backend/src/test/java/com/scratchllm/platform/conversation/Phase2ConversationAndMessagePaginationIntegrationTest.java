package com.scratchllm.platform.conversation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.patch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.Cookie;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

/**
 * Day013 & Day014 integration tests:
 * - Creator-private conversation isolation (User B cannot read, rename, list messages, or delete User A's conversation
 *   even when both belong to the same workspace)
 * - Optimistic locking on conversation rename
 * - Keyset pagination on conversations and messages
 */
@SpringBootTest
@AutoConfigureMockMvc
class Phase2ConversationAndMessagePaginationIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Test
    void enforcesCreatorPrivateIsolationEvenWithinSharedWorkspaceAndSupportsKeysetPagination() throws Exception {
        AuthClient alice = registerAndLogin("alice-conv-" + UUID.randomUUID() + "@example.com", "Alice Shared Lab");
        AuthClient bob = registerAndLogin("bob-conv-" + UUID.randomUUID() + "@example.com", "Bob Personal Lab");

        // Add Bob as a MEMBER of Alice's workspace so both belong to the same workspace
        jdbcTemplate.update(
            "INSERT INTO memberships (id, workspace_id, user_id, role) VALUES (?, ?, ?, 'MEMBER')",
            UUID.randomUUID(), alice.workspaceId(), bob.userId()
        );

        // Alice creates 3 conversations in the shared workspace
        List<String> aliceConvIds = new ArrayList<>();
        for (int i = 1; i <= 3; i++) {
            MvcResult created = mockMvc.perform(post("/api/v1/workspaces/" + alice.workspaceId() + "/conversations")
                    .cookie(alice.xsrfCookie(), alice.sessionCookie())
                    .header("X-CSRF-TOKEN", alice.csrfToken())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(Map.of("title", "Alice Private Chat " + i))))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.version").value(0))
                .andReturn();
            JsonNode convJson = objectMapper.readTree(created.getResponse().getContentAsString());
            aliceConvIds.add(convJson.get("conversation_id").asText());
        }

        String targetConvId = aliceConvIds.getFirst();

        // Bob (a member of the same workspace!) cannot read, rename, list messages, or delete Alice's conversation
        mockMvc.perform(get("/api/v1/conversations/" + targetConvId)
                .cookie(bob.sessionCookie()))
            .andExpect(status().isNotFound())
            .andExpect(jsonPath("$.error_code").value("CONVERSATION_NOT_FOUND"));

        mockMvc.perform(patch("/api/v1/conversations/" + targetConvId)
                .cookie(bob.xsrfCookie(), bob.sessionCookie())
                .header("X-CSRF-TOKEN", bob.csrfToken())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of("title", "Bob Hijack", "version", 0))))
            .andExpect(status().isNotFound());

        mockMvc.perform(get("/api/v1/conversations/" + targetConvId + "/messages")
                .cookie(bob.sessionCookie()))
            .andExpect(status().isNotFound());

        mockMvc.perform(delete("/api/v1/conversations/" + targetConvId)
                .cookie(bob.xsrfCookie(), bob.sessionCookie())
                .header("X-CSRF-TOKEN", bob.csrfToken()))
            .andExpect(status().isNotFound());

        // Bob's conversation list in Alice's workspace is empty because conversations are creator-private
        mockMvc.perform(get("/api/v1/workspaces/" + alice.workspaceId() + "/conversations")
                .cookie(bob.sessionCookie()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.items.length()").value(0))
            .andExpect(jsonPath("$.has_more").value(false));

        // Alice renames her conversation with optimistic lock version=0 -> version becomes 1
        mockMvc.perform(patch("/api/v1/conversations/" + targetConvId)
                .cookie(alice.xsrfCookie(), alice.sessionCookie())
                .header("X-CSRF-TOKEN", alice.csrfToken())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of("title", "Alice Renamed Chat", "version", 0))))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.title").value("Alice Renamed Chat"))
            .andExpect(jsonPath("$.version").value(1));

        // Stale version=0 rename attempt fails with 409 VERSION_CONFLICT
        mockMvc.perform(patch("/api/v1/conversations/" + targetConvId)
                .cookie(alice.xsrfCookie(), alice.sessionCookie())
                .header("X-CSRF-TOKEN", alice.csrfToken())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of("title", "Stale Update", "version", 0))))
            .andExpect(status().isConflict())
            .andExpect(jsonPath("$.error_code").value("VERSION_CONFLICT"));

        // Keyset pagination on Alice's 3 conversations with limit=2
        MvcResult page1Res = mockMvc.perform(get("/api/v1/workspaces/" + alice.workspaceId() + "/conversations")
                .param("limit", "2")
                .cookie(alice.sessionCookie()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.items.length()").value(2))
            .andExpect(jsonPath("$.has_more").value(true))
            .andExpect(jsonPath("$.next_cursor").isNotEmpty())
            .andReturn();

        String nextCursor = objectMapper.readTree(page1Res.getResponse().getContentAsString())
            .get("next_cursor").asText();

        mockMvc.perform(get("/api/v1/workspaces/" + alice.workspaceId() + "/conversations")
                .param("limit", "2")
                .param("cursor", nextCursor)
                .cookie(alice.sessionCookie()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.items.length()").value(1))
            .andExpect(jsonPath("$.has_more").value(false));

        // Insert 3 ordered messages into Alice's target conversation and test message keyset pagination
        UUID convUuid = UUID.fromString(targetConvId);
        for (int m = 1; m <= 3; m++) {
            jdbcTemplate.update(
                """
                INSERT INTO messages (id, conversation_id, role, content, status, message_order)
                VALUES (?, ?, ?, ?, 'COMMITTED', ?)
                """,
                UUID.randomUUID(), convUuid, (m % 2 == 1 ? "USER" : "ASSISTANT"), "Message #" + m, m
            );
        }

        MvcResult msgPage1 = mockMvc.perform(get("/api/v1/conversations/" + targetConvId + "/messages")
                .param("limit", "2")
                .cookie(alice.sessionCookie()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.items.length()").value(2))
            .andExpect(jsonPath("$.items[0].message_order").value(1))
            .andExpect(jsonPath("$.items[1].message_order").value(2))
            .andExpect(jsonPath("$.has_more").value(true))
            .andReturn();

        String msgCursor = objectMapper.readTree(msgPage1.getResponse().getContentAsString())
            .get("next_cursor").asText();

        mockMvc.perform(get("/api/v1/conversations/" + targetConvId + "/messages")
                .param("limit", "2")
                .param("cursor", msgCursor)
                .cookie(alice.sessionCookie()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.items.length()").value(1))
            .andExpect(jsonPath("$.items[0].message_order").value(3))
            .andExpect(jsonPath("$.has_more").value(false));
    }

    private AuthClient registerAndLogin(String email, String workspaceName) throws Exception {
        MvcResult csrfRes = mockMvc.perform(get("/auth/csrf")).andExpect(status().isOk()).andReturn();
        String csrfToken = objectMapper.readTree(csrfRes.getResponse().getContentAsString()).get("token").asText();
        Cookie xsrfCookie = csrfRes.getResponse().getCookie("XSRF-TOKEN");

        MvcResult regRes = mockMvc.perform(post("/auth/register")
                .cookie(xsrfCookie)
                .header("X-CSRF-TOKEN", csrfToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of(
                    "email", email,
                    "password", "StrongPass#2026!",
                    "workspace_name", workspaceName
                ))))
            .andExpect(status().isCreated())
            .andReturn();

        JsonNode profile = objectMapper.readTree(regRes.getResponse().getContentAsString());
        UUID userId = UUID.fromString(profile.get("user_id").asText());
        UUID workspaceId = UUID.fromString(profile.get("workspaces").get(0).get("workspace_id").asText());

        MvcResult loginRes = mockMvc.perform(post("/auth/login")
                .cookie(xsrfCookie)
                .header("X-CSRF-TOKEN", csrfToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of(
                    "email", email,
                    "password", "StrongPass#2026!"
                ))))
            .andExpect(status().isOk())
            .andReturn();

        Cookie sessionCookie = loginRes.getResponse().getCookie("SESSION");
        return new AuthClient(userId, workspaceId, csrfToken, xsrfCookie, sessionCookie);
    }

    private record AuthClient(
        UUID userId,
        UUID workspaceId,
        String csrfToken,
        Cookie xsrfCookie,
        Cookie sessionCookie
    ) {}
}
