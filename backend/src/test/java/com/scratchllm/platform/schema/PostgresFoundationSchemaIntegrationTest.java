package com.scratchllm.platform.schema;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.sql.DriverManager;
import java.sql.SQLException;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.UUID;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.TransactionDefinition;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionTemplate;

/**
 * Day004 & Day006 real PostgreSQL integration test:
 * Verifies Flyway migrations, pgvector/pgcrypto extensions, seeded capacity/model_revisions,
 * partial unique indexes (uq_active_run_per_conversation, uq_assistant_msg_per_run),
 * composite workspace-membership foreign key, and database credential isolation.
 */
@SpringBootTest
@Transactional
class PostgresFoundationSchemaIntegrationTest {

    @Autowired
    private JdbcTemplate jdbcTemplate;

    private void executeInSavepoint(Runnable action) {
        jdbcTemplate.execute("SAVEPOINT sp_constraint_test");
        try {
            action.run();
            jdbcTemplate.execute("RELEASE SAVEPOINT sp_constraint_test");
        } catch (RuntimeException ex) {
            jdbcTemplate.execute("ROLLBACK TO SAVEPOINT sp_constraint_test");
            jdbcTemplate.execute("RELEASE SAVEPOINT sp_constraint_test");
            throw ex;
        }
    }

    @Test
    void flywayMigrationAppliesExtensionsAndSeedsBaselineRows() {
        List<String> extensions = jdbcTemplate.queryForList(
            "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pgcrypto') ORDER BY extname",
            String.class
        );
        assertThat(extensions).containsExactly("pgcrypto", "vector");

        Integer maxWaitingRuns = jdbcTemplate.queryForObject(
            "SELECT max_waiting_runs FROM system_capacity WHERE id = 1",
            Integer.class
        );
        assertThat(maxWaitingRuns).isEqualTo(8);

        Integer seededModels = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM model_revisions WHERE revision_tag IN ('fake-deterministic-r1', 'scratch-tiny-r0')",
            Integer.class
        );
        assertThat(seededModels).isEqualTo(2);
    }

    @Test
    void enforcesPartialUniqueIndexesAndWorkspaceMembershipConstraints() {
        UUID ownerId = UUID.randomUUID();
        UUID outsiderId = UUID.randomUUID();
        UUID workspaceId = UUID.randomUUID();
        UUID conversationId = UUID.randomUUID();

        jdbcTemplate.update(
            "INSERT INTO users (id, email, password_hash, status) VALUES (?, ?, ?, 'ACTIVE')",
            ownerId, "owner-" + ownerId + "@example.com", "$2a$12$abcdefghijklmnopqrstuv"
        );
        jdbcTemplate.update(
            "INSERT INTO users (id, email, password_hash, status) VALUES (?, ?, ?, 'ACTIVE')",
            outsiderId, "outsider-" + outsiderId + "@example.com", "$2a$12$abcdefghijklmnopqrstuv"
        );
        jdbcTemplate.update(
            "INSERT INTO workspaces (id, name, owner_user_id) VALUES (?, 'Primary Workspace', ?)",
            workspaceId, ownerId
        );
        jdbcTemplate.update(
            "INSERT INTO memberships (id, workspace_id, user_id, role) VALUES (?, ?, ?, 'OWNER')",
            UUID.randomUUID(), workspaceId, ownerId
        );

        // Outsider without membership cannot own a conversation in workspaceId (PostgreSQL savepoint keeps tx valid)
        assertThatThrownBy(() -> executeInSavepoint(() -> jdbcTemplate.update(
            "INSERT INTO conversations (id, workspace_id, owner_user_id, title) VALUES (?, ?, ?, 'Unauthorized')",
            UUID.randomUUID(), workspaceId, outsiderId
        ))).isInstanceOf(DataIntegrityViolationException.class);

        // Authorized workspace member can create conversation
        jdbcTemplate.update(
            "INSERT INTO conversations (id, workspace_id, owner_user_id, title) VALUES (?, ?, ?, 'Valid Chat')",
            conversationId, workspaceId, ownerId
        );

        UUID firstRunId = UUID.randomUUID();
        OffsetDateTime deadline = OffsetDateTime.now(ZoneOffset.UTC).plusMinutes(2);

        jdbcTemplate.update(
            """
            INSERT INTO generation_runs (
                id, workspace_id, conversation_id, user_id, status, model_revision,
                prompt_snapshot, prompt_snapshot_hash, sampling_json, queue_deadline_at, execution_deadline_at
            ) VALUES (?, ?, ?, ?, 'QUEUED', 'fake-deterministic-r1', 'Hello', 'hash1', '{}'::jsonb, ?, ?)
            """,
            firstRunId, workspaceId, conversationId, ownerId, deadline, deadline
        );

        // Partial unique index uq_active_run_per_conversation blocks a second active run in the same conversation
        assertThatThrownBy(() -> executeInSavepoint(() -> jdbcTemplate.update(
            """
            INSERT INTO generation_runs (
                id, workspace_id, conversation_id, user_id, status, model_revision,
                prompt_snapshot, prompt_snapshot_hash, sampling_json, queue_deadline_at, execution_deadline_at
            ) VALUES (?, ?, ?, ?, 'RUNNING', 'fake-deterministic-r1', 'Concurrent', 'hash2', '{}'::jsonb, ?, ?)
            """,
            UUID.randomUUID(), workspaceId, conversationId, ownerId, deadline, deadline
        ))).isInstanceOf(DataIntegrityViolationException.class);

        // Partial unique index uq_assistant_msg_per_run blocks two ASSISTANT messages for the same run
        jdbcTemplate.update(
            """
            INSERT INTO messages (id, conversation_id, run_id, role, content, status, message_order)
            VALUES (?, ?, ?, 'ASSISTANT', 'First answer', 'COMMITTED', 1)
            """,
            UUID.randomUUID(), conversationId, firstRunId
        );

        assertThatThrownBy(() -> executeInSavepoint(() -> jdbcTemplate.update(
            """
            INSERT INTO messages (id, conversation_id, run_id, role, content, status, message_order)
            VALUES (?, ?, ?, 'ASSISTANT', 'Duplicate answer', 'COMMITTED', 2)
            """,
            UUID.randomUUID(), conversationId, firstRunId
        ))).isInstanceOf(DataIntegrityViolationException.class);

        // Once the first run transitions to COMPLETED, a new QUEUED run in the same conversation is allowed
        jdbcTemplate.update(
            "UPDATE generation_runs SET status = 'COMPLETED', completed_at = clock_timestamp() WHERE id = ?",
            firstRunId
        );
        int insertedSecondRun = jdbcTemplate.update(
            """
            INSERT INTO generation_runs (
                id, workspace_id, conversation_id, user_id, status, model_revision,
                prompt_snapshot, prompt_snapshot_hash, sampling_json, queue_deadline_at, execution_deadline_at
            ) VALUES (?, ?, ?, ?, 'QUEUED', 'fake-deterministic-r1', 'Follow-up', 'hash3', '{}'::jsonb, ?, ?)
            """,
            UUID.randomUUID(), workspaceId, conversationId, ownerId, deadline, deadline
        );
        assertThat(insertedSecondRun).isEqualTo(1);
    }

    @Test
    void enforcesDatabaseRoleIsolationAcrossAppMlflowAndAirflowDatabases() {
        assertThatThrownBy(() -> DriverManager.getConnection(
            "jdbc:postgresql://127.0.0.1:54322/mlflow_db",
            "llm_app_user",
            "llm_app_local_pass"
        )).isInstanceOf(SQLException.class);

        assertThatThrownBy(() -> DriverManager.getConnection(
            "jdbc:postgresql://127.0.0.1:54322/airflow_db",
            "llm_app_user",
            "llm_app_local_pass"
        )).isInstanceOf(SQLException.class);
    }
}
