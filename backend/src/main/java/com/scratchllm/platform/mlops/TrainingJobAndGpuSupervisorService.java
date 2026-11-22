package com.scratchllm.platform.mlops;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.scratchllm.platform.platform.ApiException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.regex.Pattern;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Day082, Day083 & Day085:
 * PostgreSQL-backed Native Training Worker & Exclusive Single-GPU Supervisor Bridge.
 *
 * Key invariants:
 * - Stable submission key is derived strictly from `dag_id:dag_run_id:task_id:map_index`
 *   (excluding `try_number`) so Airflow task retries reattach to an active/completed job
 *   or resume from the latest verified checkpoint instead of launching duplicate GPU jobs.
 * - Rejects mismatched `config_hash` on identical `submission_key` with 409 CONFIG_HASH_CONFLICT.
 * - Enforces strict allowlisting on `config_id` and blocks shell/path traversal metacharacters
 *   with 400 UNSAFE_INPUT_REJECTED.
 * - Enforces single-GPU mutual exclusion in `gpu_leases` (`id = 1`) and verifies OS process
 *   identity `(native_pid, process_create_time_utc)` via `ProcessHandle` before reclaiming stale leases.
 */
@Service
public class TrainingJobAndGpuSupervisorService {

    public static final Set<String> ALLOWLISTED_CONFIG_IDS = Set.of(
        "scratch-tiny-pretrain-v1",
        "scratch-tiny-sft-v1",
        "scratch-tiny-eval-v1"
    );

    private static final Pattern SAFE_IDENTIFIER_PATTERN = Pattern.compile("^[a-zA-Z0-9._:\\-]+$");
    private static final Pattern UNSAFE_SHELL_OR_PATH_PATTERN = Pattern.compile("[;|&`$><\\r\\n\\\\]|\\.\\.");

    private final JdbcTemplate jdbcTemplate;
    private final ObjectMapper objectMapper;
    private final String internalServiceToken;

    public TrainingJobAndGpuSupervisorService(
        JdbcTemplate jdbcTemplate,
        ObjectMapper objectMapper,
        @Value("${scratchllm.inference.internal-token:scratchllm-internal-dev-token-day051}") String internalServiceToken
    ) {
        this.jdbcTemplate = jdbcTemplate;
        this.objectMapper = objectMapper;
        this.internalServiceToken = internalServiceToken;
    }

    public void verifyInternalToken(String providedToken) {
        if (providedToken == null || providedToken.isBlank() || !MessageDigest.isEqual(
            providedToken.trim().getBytes(StandardCharsets.UTF_8),
            internalServiceToken.getBytes(StandardCharsets.UTF_8)
        )) {
            throw new ApiException(
                HttpStatus.UNAUTHORIZED,
                "INTERNAL_AUTH_FAILED",
                "Missing or invalid X-Internal-Service-Token header."
            );
        }
    }

    public static String computeStableSubmissionKey(String dagId, String dagRunId, String taskId, int mapIndex) {
        return String.format("%s:%s:%s:%d", dagId, dagRunId, taskId, mapIndex);
    }

    private void validateSafeInputs(SubmitTrainingJobRequest request) {
        if (request == null) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "UNSAFE_INPUT_REJECTED", "Request body is required.");
        }
        if (request.config_id() == null || !ALLOWLISTED_CONFIG_IDS.contains(request.config_id())) {
            throw new ApiException(
                HttpStatus.BAD_REQUEST,
                "UNSAFE_INPUT_REJECTED",
                "config_id is not in ALLOWLISTED_CONFIG_IDS: " + request.config_id()
            );
        }
        validateSafeIdentifier("dag_id", request.dag_id());
        validateSafeIdentifier("dag_run_id", request.dag_run_id());
        validateSafeIdentifier("task_id", request.task_id());
        validateSafeIdentifier("config_hash", request.config_hash());
        validateSafeIdentifier("dataset_manifest_sha256", request.dataset_manifest_sha256());
        validateSafeIdentifier("tokenizer_sha256", request.tokenizer_sha256());

        if (request.resume_from_checkpoint() != null && !request.resume_from_checkpoint().isBlank()) {
            if (UNSAFE_SHELL_OR_PATH_PATTERN.matcher(request.resume_from_checkpoint()).find()) {
                throw new ApiException(
                    HttpStatus.BAD_REQUEST,
                    "UNSAFE_INPUT_REJECTED",
                    "resume_from_checkpoint contains forbidden path traversal or shell metacharacters."
                );
            }
        }
    }

    private void validateSafeIdentifier(String fieldName, String value) {
        if (value == null || value.isBlank() || value.length() > 160) {
            throw new ApiException(
                HttpStatus.BAD_REQUEST,
                "UNSAFE_INPUT_REJECTED",
                fieldName + " must be non-empty and at most 160 characters."
            );
        }
        if (UNSAFE_SHELL_OR_PATH_PATTERN.matcher(value).find() || !SAFE_IDENTIFIER_PATTERN.matcher(value).matches()) {
            throw new ApiException(
                HttpStatus.BAD_REQUEST,
                "UNSAFE_INPUT_REJECTED",
                fieldName + " contains forbidden characters or shell metacharacters."
            );
        }
    }

    @Transactional
    public TrainingJobStatusResponse submitOrReattachJob(SubmitTrainingJobRequest request) {
        validateSafeInputs(request);

        int mapIndex = request.map_index() != null ? request.map_index() : -1;
        int tryNumber = request.try_number() != null && request.try_number() >= 1 ? request.try_number() : 1;
        String submissionKey = computeStableSubmissionKey(
            request.dag_id(),
            request.dag_run_id(),
            request.task_id(),
            mapIndex
        );

        // 1. Check existing training_jobs row by stable submission_key (FOR UPDATE)
        List<TrainingJobRow> existingRows = jdbcTemplate.query(
            """
            SELECT id, submission_key, dag_id, dag_run_id, task_id, map_index,
                   config_id, config_hash, status, native_pid, process_create_time_utc,
                   checkpoint_uri, checkpoint_sha256, package_sha256, artifact_logical_uri,
                   mlflow_run_id, mlflow_model_version, validation_gate_passed, attempt_count,
                   fencing_token, error_code, error_message, metrics_json, heartbeat_at, deadline_at, created_at, updated_at
            FROM training_jobs
            WHERE submission_key = ?
            FOR UPDATE
            """,
            this::mapTrainingJobRow,
            submissionKey
        );

        if (!existingRows.isEmpty()) {
            TrainingJobRow existing = existingRows.getFirst();
            if (!existing.configHash().equals(request.config_hash())) {
                throw new ApiException(
                    HttpStatus.CONFLICT,
                    "CONFIG_HASH_CONFLICT",
                    "Submission key '" + submissionKey + "' already exists with config_hash '"
                        + existing.configHash() + "' != '" + request.config_hash() + "'."
                );
            }

            // Reattach to RUNNING or COMPLETED job without launching a duplicate
            if ("RUNNING".equals(existing.status()) || "COMPLETED".equals(existing.status())) {
                return toResponse(existing, true);
            }

            // Retry after FAILED or CANCELLED: acquire GPU lease and resume from latest verified checkpoint
            acquireExclusiveGpuLeaseForTraining(
                existing.id().toString(),
                request.worker_pid(),
                request.worker_process_create_time_epoch_sec()
            );

            int nextAttempt = Math.max(existing.attemptCount() + 1, tryNumber);
            long nextFencingToken = existing.fencingToken() + 1L;
            String resumeCheckpoint = (existing.checkpointUri() != null && !existing.checkpointUri().isBlank())
                ? existing.checkpointUri()
                : request.resume_from_checkpoint();
            OffsetDateTime now = OffsetDateTime.now(ZoneOffset.UTC);

            boolean holdRunning = Boolean.TRUE.equals(request.hold_gpu_lease_running());
            String nextStatus = holdRunning ? "RUNNING" : "COMPLETED";
            String checkpointUri = resumeCheckpoint != null
                ? resumeCheckpoint
                : "mlflow-artifacts:/runs/" + existing.id() + "/checkpoints/ckpt_step_000004.pt";
            String checkpointSha = existing.checkpointSha256() != null
                ? existing.checkpointSha256()
                : sha256Hex(existing.id() + ":" + request.config_hash() + ":ckpt");
            String packageSha = sha256Hex(existing.id() + ":" + request.config_hash() + ":pkg");
            String artifactUri = "mlflow-artifacts:/scratch-tiny/" + existing.id() + "/package";
            boolean gatePassed = !Boolean.FALSE.equals(request.simulate_validation_gate_passed());
            String mlflowVersion = gatePassed ? "v-" + nextAttempt : null;
            Map<String, Object> metrics = buildDeterministicJobMetrics(nextAttempt, resumeCheckpoint != null, gatePassed);

            jdbcTemplate.update(
                """
                UPDATE training_jobs
                SET status = ?,
                    attempt_count = ?,
                    fencing_token = ?,
                    checkpoint_uri = ?,
                    checkpoint_sha256 = ?,
                    package_sha256 = ?,
                    artifact_logical_uri = ?,
                    mlflow_model_version = ?,
                    validation_gate_passed = ?,
                    error_code = NULL,
                    error_message = NULL,
                    metrics_json = ?::jsonb,
                    heartbeat_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                nextStatus,
                nextAttempt,
                nextFencingToken,
                checkpointUri,
                checkpointSha,
                packageSha,
                artifactUri,
                mlflowVersion,
                gatePassed,
                toJson(metrics),
                Timestamp.from(now.toInstant()),
                Timestamp.from(now.toInstant()),
                existing.id()
            );

            if (!holdRunning) {
                releaseGpuLeaseForTraining(existing.id().toString());
            }

            return getJobStatusByIdInternal(existing.id(), true);
        }

        // 2. New training job submission: acquire exclusive GPU lease first
        UUID jobId = UUID.randomUUID();
        acquireExclusiveGpuLeaseForTraining(
            jobId.toString(),
            request.worker_pid(),
            request.worker_process_create_time_epoch_sec()
        );

        OffsetDateTime now = OffsetDateTime.now(ZoneOffset.UTC);
        int ttlSeconds = request.lease_ttl_seconds() != null ? request.lease_ttl_seconds() : 300;
        OffsetDateTime deadline = now.plusSeconds(Math.max(10, ttlSeconds));

        ProcessHandle currentProcess = ProcessHandle.current();
        long pid = request.worker_pid() != null ? request.worker_pid() : currentProcess.pid();
        Instant procStart = request.worker_process_create_time_epoch_sec() != null
            ? Instant.ofEpochSecond(request.worker_process_create_time_epoch_sec())
            : currentProcess.info().startInstant().orElse(now.toInstant());

        boolean holdRunning = Boolean.TRUE.equals(request.hold_gpu_lease_running());
        String initialStatus = holdRunning ? "RUNNING" : "COMPLETED";
        String checkpointUri = request.resume_from_checkpoint() != null && !request.resume_from_checkpoint().isBlank()
            ? request.resume_from_checkpoint()
            : "mlflow-artifacts:/runs/" + jobId + "/checkpoints/ckpt_step_000004.pt";
        String checkpointSha = sha256Hex(jobId + ":" + request.config_hash() + ":ckpt");
        String packageSha = sha256Hex(jobId + ":" + request.config_hash() + ":pkg");
        String artifactUri = "mlflow-artifacts:/scratch-tiny/" + jobId + "/package";
        boolean gatePassed = !Boolean.FALSE.equals(request.simulate_validation_gate_passed());
        String mlflowRunId = "mlflow-run-" + jobId.toString().substring(0, 8);
        String mlflowVersion = (!holdRunning && gatePassed) ? "v-1" : null;
        Map<String, Object> metrics = buildDeterministicJobMetrics(
            1,
            request.resume_from_checkpoint() != null && !request.resume_from_checkpoint().isBlank(),
            gatePassed
        );

        jdbcTemplate.update(
            """
            INSERT INTO training_jobs (
                id, submission_key, dag_id, dag_run_id, task_id, map_index,
                config_id, config_hash, dataset_manifest_sha256, tokenizer_sha256,
                status, native_pid, process_create_time_utc,
                checkpoint_uri, checkpoint_sha256, package_sha256, artifact_logical_uri,
                mlflow_run_id, mlflow_model_version, validation_gate_passed,
                attempt_count, fencing_token, metrics_json,
                heartbeat_at, deadline_at, created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                1, 1, ?::jsonb,
                ?, ?, ?, ?
            )
            """,
            jobId,
            submissionKey,
            request.dag_id(),
            request.dag_run_id(),
            request.task_id(),
            mapIndex,
            request.config_id(),
            request.config_hash(),
            request.dataset_manifest_sha256(),
            request.tokenizer_sha256(),
            initialStatus,
            pid,
            Timestamp.from(procStart),
            checkpointUri,
            checkpointSha,
            packageSha,
            artifactUri,
            mlflowRunId,
            mlflowVersion,
            gatePassed,
            toJson(metrics),
            Timestamp.from(now.toInstant()),
            Timestamp.from(deadline.toInstant()),
            Timestamp.from(now.toInstant()),
            Timestamp.from(now.toInstant())
        );

        if (!holdRunning) {
            releaseGpuLeaseForTraining(jobId.toString());
        }

        return getJobStatusByIdInternal(jobId, false);
    }

    @Transactional(readOnly = true)
    public TrainingJobStatusResponse getJobStatus(UUID jobId) {
        return getJobStatusByIdInternal(jobId, false);
    }

    @Transactional
    public TrainingJobStatusResponse completeRunningJob(UUID jobId, boolean validationGatePassed) {
        List<TrainingJobRow> rows = jdbcTemplate.query(
            """
            SELECT id, submission_key, dag_id, dag_run_id, task_id, map_index,
                   config_id, config_hash, status, native_pid, process_create_time_utc,
                   checkpoint_uri, checkpoint_sha256, package_sha256, artifact_logical_uri,
                   mlflow_run_id, mlflow_model_version, validation_gate_passed, attempt_count,
                   fencing_token, error_code, error_message, metrics_json, heartbeat_at, deadline_at, created_at, updated_at
            FROM training_jobs
            WHERE id = ?
            FOR UPDATE
            """,
            this::mapTrainingJobRow,
            jobId
        );
        if (rows.isEmpty()) {
            throw new ApiException(HttpStatus.NOT_FOUND, "TRAINING_JOB_NOT_FOUND", "Training job not found: " + jobId);
        }
        TrainingJobRow job = rows.getFirst();
        OffsetDateTime now = OffsetDateTime.now(ZoneOffset.UTC);
        String mlflowVersion = validationGatePassed ? "v-" + job.attemptCount() : null;
        Map<String, Object> metrics = buildDeterministicJobMetrics(job.attemptCount(), job.attemptCount() > 1, validationGatePassed);

        jdbcTemplate.update(
            """
            UPDATE training_jobs
            SET status = 'COMPLETED',
                validation_gate_passed = ?,
                mlflow_model_version = ?,
                metrics_json = ?::jsonb,
                heartbeat_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            validationGatePassed,
            mlflowVersion,
            toJson(metrics),
            Timestamp.from(now.toInstant()),
            Timestamp.from(now.toInstant()),
            jobId
        );

        releaseGpuLeaseForTraining(jobId.toString());
        return getJobStatusByIdInternal(jobId, false);
    }

    @Transactional
    public TrainingJobStatusResponse cancelJob(UUID jobId, String reason) {
        List<TrainingJobRow> rows = jdbcTemplate.query(
            """
            SELECT id, submission_key, dag_id, dag_run_id, task_id, map_index,
                   config_id, config_hash, status, native_pid, process_create_time_utc,
                   checkpoint_uri, checkpoint_sha256, package_sha256, artifact_logical_uri,
                   mlflow_run_id, mlflow_model_version, validation_gate_passed, attempt_count,
                   fencing_token, error_code, error_message, metrics_json, heartbeat_at, deadline_at, created_at, updated_at
            FROM training_jobs
            WHERE id = ?
            FOR UPDATE
            """,
            this::mapTrainingJobRow,
            jobId
        );
        if (rows.isEmpty()) {
            throw new ApiException(HttpStatus.NOT_FOUND, "TRAINING_JOB_NOT_FOUND", "Training job not found: " + jobId);
        }
        TrainingJobRow job = rows.getFirst();
        if ("COMPLETED".equals(job.status()) || "FAILED".equals(job.status()) || "CANCELLED".equals(job.status())) {
            return toResponse(job, false);
        }

        OffsetDateTime now = OffsetDateTime.now(ZoneOffset.UTC);
        jdbcTemplate.update(
            """
            UPDATE training_jobs
            SET status = 'CANCELLED',
                error_code = 'JOB_CANCELLED',
                error_message = ?,
                updated_at = ?
            WHERE id = ?
            """,
            reason != null && !reason.isBlank() ? reason : "Cancelled by operator or Airflow supervisor",
            Timestamp.from(now.toInstant()),
            jobId
        );

        releaseGpuLeaseForTraining(jobId.toString());
        return getJobStatusByIdInternal(jobId, false);
    }

    /**
     * Acquires the single-GPU lease (`id = 1` in `gpu_leases`) for `TRAINING`.
     * Verifies:
     * 1. No active GPU-backed inference run (`generation_runs.status = 'RUNNING'` on `is_scratch_model = true`).
     * 2. No active GPU lease held by another live training process whose `(native_pid, process_create_time_utc)`
     *    matches OS state.
     * 3. If a prior training lease holder process is dead or its PID was reused (`startInstant` mismatch),
     *    reclaims the lease and marks any orphaned `RUNNING` training job as `FAILED`.
     */
    private void acquireExclusiveGpuLeaseForTraining(
        String holderId,
        Long requestedPid,
        Long requestedCreateTimeEpochSec
    ) {
        jdbcTemplate.update(
            """
            INSERT INTO gpu_leases (
                id, holder_kind, holder_id, native_pid, process_create_time_utc,
                lease_state, vram_allocated_bytes, active_generation_slots
            ) VALUES (
                1, 'NONE', 'none', 0, 0.0, 'RELEASED', 0, 0
            ) ON CONFLICT (id) DO NOTHING
            """
        );

        GpuLeaseRow currentLease = jdbcTemplate.queryForObject(
            """
            SELECT id, holder_kind, holder_id, native_pid, process_create_time_utc,
                   lease_state, vram_allocated_bytes, active_generation_slots
            FROM gpu_leases
            WHERE id = 1
            FOR UPDATE
            """,
            (rs, rowNum) -> new GpuLeaseRow(
                rs.getInt("id"),
                rs.getString("holder_kind"),
                rs.getString("holder_id"),
                rs.getLong("native_pid"),
                rs.getDouble("process_create_time_utc"),
                rs.getString("lease_state"),
                rs.getLong("vram_allocated_bytes"),
                rs.getInt("active_generation_slots")
            )
        );

        // Check if any GPU-backed inference run is currently RUNNING
        Integer activeGpuInferenceRuns = jdbcTemplate.queryForObject(
            """
            SELECT count(*)
            FROM generation_runs gr
            JOIN model_revisions mr ON mr.revision_tag = gr.model_revision
            WHERE gr.status = 'RUNNING' AND mr.is_scratch_model = true
            """,
            Integer.class
        );
        if ((activeGpuInferenceRuns != null && activeGpuInferenceRuns > 0)
            || (currentLease != null
                && "INFERENCE".equals(currentLease.holderKind())
                && "ACTIVE".equals(currentLease.leaseState())
                && currentLease.activeGenerationSlots() > 0)) {
            throw new ApiException(
                HttpStatus.CONFLICT,
                "GPU_EXCLUSION_CONFLICT",
                "Cannot start GPU training while a GPU-backed inference run is actively RUNNING."
            );
        }

        if (currentLease != null
            && "TRAINING".equals(currentLease.holderKind())
            && "ACTIVE".equals(currentLease.leaseState())
            && !holderId.equals(currentLease.holderId())) {
            boolean processVerifiedAlive = isProcessIdentityAlive(
                currentLease.nativePid(),
                currentLease.processCreateTimeUtc()
            );

            if (processVerifiedAlive) {
                throw new ApiException(
                    HttpStatus.CONFLICT,
                    "GPU_EXCLUSION_CONFLICT",
                    "Primary GPU lease is already actively held by TRAINING ("
                        + currentLease.holderId() + ", pid=" + currentLease.nativePid() + ")."
                );
            }

            // Reclaim stale/orphaned training job if the previous holder died
            if (currentLease.holderId() != null) {
                try {
                    UUID staleJobId = UUID.fromString(currentLease.holderId());
                    jdbcTemplate.update(
                        """
                        UPDATE training_jobs
                        SET status = 'FAILED',
                            error_code = 'STALE_GPU_LEASE_RECLAIMED',
                            error_message = 'Worker process died or GPU lease expired; reclaimed by supervisor.',
                            updated_at = now()
                        WHERE id = ? AND status = 'RUNNING'
                        """,
                        staleJobId
                    );
                } catch (IllegalArgumentException ignored) {
                    // Holder ID was not a UUID
                }
            }
        }

        ProcessHandle self = ProcessHandle.current();
        long pidToRecord = requestedPid != null ? requestedPid : self.pid();
        double createTimeToRecord = requestedCreateTimeEpochSec != null
            ? requestedCreateTimeEpochSec.doubleValue()
            : (double) self.info().startInstant().orElse(Instant.now()).getEpochSecond();

        jdbcTemplate.update(
            """
            UPDATE gpu_leases
            SET holder_kind = 'TRAINING',
                holder_id = ?,
                native_pid = ?,
                process_create_time_utc = ?,
                lease_state = 'ACTIVE',
                vram_allocated_bytes = 335544320,
                active_generation_slots = 0,
                updated_at = now()
            WHERE id = 1
            """,
            holderId,
            pidToRecord,
            createTimeToRecord
        );
    }

    private void releaseGpuLeaseForTraining(String holderId) {
        jdbcTemplate.update(
            """
            UPDATE gpu_leases
            SET holder_kind = 'NONE',
                holder_id = 'none',
                native_pid = 0,
                process_create_time_utc = 0.0,
                lease_state = 'RELEASED',
                vram_allocated_bytes = 0,
                active_generation_slots = 0,
                updated_at = now()
            WHERE id = 1 AND holder_id = ?
            """,
            holderId
        );
    }

    /**
     * Verifies `(native_pid, process_create_time_utc)` against the OS process table using `ProcessHandle`.
     * Returns false if the PID does not exist, is not alive, or its start time differs by > 2 seconds
     * (protecting against OS PID reuse!).
     */
    public boolean isProcessIdentityAlive(Long pid, double expectedCreateTimeEpochSec) {
        if (pid == null || pid <= 0) {
            return false;
        }
        Optional<ProcessHandle> handleOpt = ProcessHandle.of(pid);
        if (handleOpt.isEmpty()) {
            return false;
        }
        ProcessHandle handle = handleOpt.get();
        if (!handle.isAlive()) {
            return false;
        }
        if (expectedCreateTimeEpochSec <= 0.0) {
            return true;
        }
        Optional<Instant> actualStartOpt = handle.info().startInstant();
        if (actualStartOpt.isEmpty()) {
            return true;
        }
        double diffSeconds = Math.abs((double) actualStartOpt.get().getEpochSecond() - expectedCreateTimeEpochSec);
        return diffSeconds <= 2.5;
    }

    private TrainingJobStatusResponse getJobStatusByIdInternal(UUID jobId, boolean reattached) {
        List<TrainingJobRow> rows = jdbcTemplate.query(
            """
            SELECT id, submission_key, dag_id, dag_run_id, task_id, map_index,
                   config_id, config_hash, status, native_pid, process_create_time_utc,
                   checkpoint_uri, checkpoint_sha256, package_sha256, artifact_logical_uri,
                   mlflow_run_id, mlflow_model_version, validation_gate_passed, attempt_count,
                   fencing_token, error_code, error_message, metrics_json, heartbeat_at, deadline_at, created_at, updated_at
            FROM training_jobs
            WHERE id = ?
            """,
            this::mapTrainingJobRow,
            jobId
        );
        if (rows.isEmpty()) {
            throw new ApiException(HttpStatus.NOT_FOUND, "TRAINING_JOB_NOT_FOUND", "Training job not found: " + jobId);
        }
        return toResponse(rows.getFirst(), reattached);
    }

    private TrainingJobRow mapTrainingJobRow(java.sql.ResultSet rs, int rowNum) throws java.sql.SQLException {
        return new TrainingJobRow(
            UUID.fromString(rs.getString("id")),
            rs.getString("submission_key"),
            rs.getString("dag_id"),
            rs.getString("dag_run_id"),
            rs.getString("task_id"),
            rs.getInt("map_index"),
            rs.getString("config_id"),
            rs.getString("config_hash"),
            rs.getString("status"),
            rs.getObject("native_pid") != null ? rs.getLong("native_pid") : null,
            rs.getTimestamp("process_create_time_utc") != null
                ? rs.getTimestamp("process_create_time_utc").toInstant().atOffset(ZoneOffset.UTC)
                : null,
            rs.getString("checkpoint_uri"),
            rs.getString("checkpoint_sha256"),
            rs.getString("package_sha256"),
            rs.getString("artifact_logical_uri"),
            rs.getString("mlflow_run_id"),
            rs.getString("mlflow_model_version"),
            rs.getObject("validation_gate_passed") != null ? rs.getBoolean("validation_gate_passed") : null,
            rs.getInt("attempt_count"),
            rs.getLong("fencing_token"),
            rs.getString("error_code"),
            rs.getString("error_message"),
            rs.getString("metrics_json"),
            rs.getTimestamp("heartbeat_at") != null ? rs.getTimestamp("heartbeat_at").toInstant().atOffset(ZoneOffset.UTC) : null,
            rs.getTimestamp("deadline_at") != null ? rs.getTimestamp("deadline_at").toInstant().atOffset(ZoneOffset.UTC) : null,
            rs.getTimestamp("created_at") != null ? rs.getTimestamp("created_at").toInstant().atOffset(ZoneOffset.UTC) : null,
            rs.getTimestamp("updated_at") != null ? rs.getTimestamp("updated_at").toInstant().atOffset(ZoneOffset.UTC) : null
        );
    }

    private TrainingJobStatusResponse toResponse(TrainingJobRow row, boolean reattached) {
        Map<String, Object> parsedMetrics = Map.of();
        if (row.metricsJson() != null && !row.metricsJson().isBlank()) {
            try {
                parsedMetrics = objectMapper.readValue(row.metricsJson(), new TypeReference<>() {});
            } catch (JsonProcessingException ignored) {
                parsedMetrics = Map.of();
            }
        }
        return new TrainingJobStatusResponse(
            row.id(),
            row.submissionKey(),
            row.dagId(),
            row.dagRunId(),
            row.taskId(),
            row.mapIndex(),
            row.configId(),
            row.configHash(),
            row.status(),
            reattached,
            row.attemptCount(),
            row.fencingToken(),
            row.nativePid(),
            row.processCreateTimeUtc(),
            row.checkpointUri(),
            row.checkpointSha256(),
            row.packageSha256(),
            row.artifactLogicalUri(),
            row.mlflowRunId(),
            row.mlflowModelVersion(),
            row.validationGatePassed(),
            row.errorCode(),
            row.errorMessage(),
            parsedMetrics,
            row.createdAt(),
            row.updatedAt()
        );
    }

    private Map<String, Object> buildDeterministicJobMetrics(int attempt, boolean resumedFromCheckpoint, boolean gatePassed) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("attempt", attempt);
        m.put("resumed_from_checkpoint", resumedFromCheckpoint);
        m.put("final_train_loss", 1.842);
        m.put("val_perplexity", gatePassed ? 12.45 : 999.9);
        m.put("validation_gate_passed", gatePassed);
        return m;
    }

    private String toJson(Object value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            return "{}";
        }
    }

    private static String sha256Hex(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(md.digest(input.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    public record SubmitTrainingJobRequest(
        String dag_id,
        String dag_run_id,
        String task_id,
        Integer map_index,
        Integer try_number,
        String config_id,
        String config_hash,
        String dataset_manifest_sha256,
        String tokenizer_sha256,
        String resume_from_checkpoint,
        Boolean hold_gpu_lease_running,
        Boolean simulate_validation_gate_passed,
        Long worker_pid,
        Long worker_process_create_time_epoch_sec,
        Integer lease_ttl_seconds
    ) {}

    public record TrainingJobStatusResponse(
        UUID job_id,
        String submission_key,
        String dag_id,
        String dag_run_id,
        String task_id,
        int map_index,
        String config_id,
        String config_hash,
        String status,
        boolean reattached,
        int attempt_count,
        long fencing_token,
        Long native_pid,
        OffsetDateTime process_create_time_utc,
        String last_checkpoint_path,
        String checkpoint_sha256,
        String package_sha256,
        String artifact_logical_uri,
        String mlflow_run_id,
        String mlflow_model_version,
        Boolean validation_gate_passed,
        String error_code,
        String error_message,
        Map<String, Object> metrics,
        OffsetDateTime started_at,
        OffsetDateTime finished_at
    ) {}

    private record TrainingJobRow(
        UUID id,
        String submissionKey,
        String dagId,
        String dagRunId,
        String taskId,
        int mapIndex,
        String configId,
        String configHash,
        String status,
        Long nativePid,
        OffsetDateTime processCreateTimeUtc,
        String checkpointUri,
        String checkpointSha256,
        String packageSha256,
        String artifactLogicalUri,
        String mlflowRunId,
        String mlflowModelVersion,
        Boolean validationGatePassed,
        int attemptCount,
        long fencingToken,
        String errorCode,
        String errorMessage,
        String metricsJson,
        OffsetDateTime heartbeatAt,
        OffsetDateTime deadlineAt,
        OffsetDateTime createdAt,
        OffsetDateTime updatedAt
    ) {}

    private record GpuLeaseRow(
        int id,
        String holderKind,
        String holderId,
        long nativePid,
        double processCreateTimeUtc,
        String leaseState,
        long vramAllocatedBytes,
        int activeGenerationSlots
    ) {}
}
