package com.scratchllm.platform.identity;

import com.scratchllm.platform.identity.IdentityDtos.LoginRequest;
import com.scratchllm.platform.identity.IdentityDtos.RegisterRequest;
import com.scratchllm.platform.identity.IdentityDtos.UserProfileResponse;
import com.scratchllm.platform.identity.IdentityDtos.WorkspaceMembershipDto;
import com.scratchllm.platform.platform.ApiException;
import java.util.List;
import java.util.Locale;
import java.util.UUID;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Day011: Account creation, BCrypt password hashing, workspace enrollment, and uniform authentication.
 */
@Service
public class IdentityService {

    private final JdbcTemplate jdbcTemplate;
    private final PasswordEncoder passwordEncoder;
    private final String dummyPasswordHash;
    private final boolean registrationEnabled;

    public IdentityService(
        JdbcTemplate jdbcTemplate,
        PasswordEncoder passwordEncoder,
        @Value("${scratchllm.auth.registration-enabled:true}") boolean registrationEnabled
    ) {
        this.jdbcTemplate = jdbcTemplate;
        this.passwordEncoder = passwordEncoder;
        this.dummyPasswordHash = passwordEncoder.encode("dummy-constant-time-mitigation-pass!");
        this.registrationEnabled = registrationEnabled;
    }

    @Transactional
    public UserProfileResponse registerUser(RegisterRequest request) {
        if (!registrationEnabled) {
            throw new ApiException(
                HttpStatus.FORBIDDEN,
                "REGISTRATION_DISABLED",
                "Self-service account enrollment is disabled by current platform policy."
            );
        }

        String normalizedEmail = normalizeEmail(request.email());
        Integer existing = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM users WHERE email = ?",
            Integer.class,
            normalizedEmail
        );
        if (existing != null && existing > 0) {
            throw new ApiException(
                HttpStatus.CONFLICT,
                "ACCOUNT_ALREADY_EXISTS",
                "Unable to complete registration with the provided email address."
            );
        }

        UUID userId = UUID.randomUUID();
        UUID workspaceId = UUID.randomUUID();
        UUID membershipId = UUID.randomUUID();
        String passwordHash = passwordEncoder.encode(request.password());

        try {
            jdbcTemplate.update(
                "INSERT INTO users (id, email, password_hash, status) VALUES (?, ?, ?, 'ACTIVE')",
                userId, normalizedEmail, passwordHash
            );
            jdbcTemplate.update(
                "INSERT INTO workspaces (id, name, owner_user_id) VALUES (?, ?, ?)",
                workspaceId, request.workspace_name().trim(), userId
            );
            jdbcTemplate.update(
                "INSERT INTO memberships (id, workspace_id, user_id, role) VALUES (?, ?, ?, 'OWNER')",
                membershipId, workspaceId, userId
            );
            jdbcTemplate.update(
                """
                INSERT INTO audit_events (id, workspace_id, actor_user_id, action, resource_type, resource_id)
                VALUES (?, ?, ?, 'USER_REGISTERED', 'USER', ?)
                """,
                UUID.randomUUID(), workspaceId, userId, userId
            );
        } catch (DataIntegrityViolationException ex) {
            throw new ApiException(
                HttpStatus.CONFLICT,
                "ACCOUNT_ALREADY_EXISTS",
                "Unable to complete registration with the provided email address."
            );
        }

        return new UserProfileResponse(
            userId,
            normalizedEmail,
            List.of(new WorkspaceMembershipDto(workspaceId, request.workspace_name().trim(), "OWNER"))
        );
    }

    @Transactional(readOnly = true)
    public UserProfileResponse authenticate(LoginRequest request) {
        String normalizedEmail = normalizeEmail(request.email());
        List<UserAuthRow> rows = jdbcTemplate.query(
            "SELECT id, email, password_hash, status FROM users WHERE email = ?",
            (rs, rowNum) -> new UserAuthRow(
                UUID.fromString(rs.getString("id")),
                rs.getString("email"),
                rs.getString("password_hash"),
                rs.getString("status")
            ),
            normalizedEmail
        );

        if (rows.isEmpty()) {
            passwordEncoder.matches(request.password(), dummyPasswordHash);
            throw invalidCredentials();
        }

        UserAuthRow user = rows.getFirst();
        boolean matches = passwordEncoder.matches(request.password(), user.passwordHash());
        if (!matches || !"ACTIVE".equals(user.status())) {
            throw invalidCredentials();
        }

        return getUserProfile(user.id());
    }

    @Transactional(readOnly = true)
    public UserProfileResponse getUserProfile(UUID userId) {
        List<String> emails = jdbcTemplate.query(
            "SELECT email FROM users WHERE id = ? AND status = 'ACTIVE'",
            (rs, rowNum) -> rs.getString("email"),
            userId
        );
        if (emails.isEmpty()) {
            throw new ApiException(HttpStatus.UNAUTHORIZED, "UNAUTHORIZED", "Account is no longer active.");
        }

        List<WorkspaceMembershipDto> workspaces = jdbcTemplate.query(
            """
            SELECT w.id AS workspace_id, w.name, m.role
            FROM memberships m
            JOIN workspaces w ON w.id = m.workspace_id
            WHERE m.user_id = ?
            ORDER BY m.created_at ASC
            """,
            (rs, rowNum) -> new WorkspaceMembershipDto(
                UUID.fromString(rs.getString("workspace_id")),
                rs.getString("name"),
                rs.getString("role")
            ),
            userId
        );

        return new UserProfileResponse(userId, emails.getFirst(), workspaces);
    }

    private static String normalizeEmail(String email) {
        return email.trim().toLowerCase(Locale.ROOT);
    }

    private static ApiException invalidCredentials() {
        return new ApiException(
            HttpStatus.UNAUTHORIZED,
            "INVALID_CREDENTIALS",
            "Invalid email or password."
        );
    }

    private record UserAuthRow(UUID id, String email, String passwordHash, String status) {}
}
