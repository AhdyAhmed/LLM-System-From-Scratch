package com.scratchllm.platform.mlops;

import com.scratchllm.platform.mlops.TrainingJobAndGpuSupervisorService.SubmitTrainingJobRequest;
import com.scratchllm.platform.mlops.TrainingJobAndGpuSupervisorService.TrainingJobStatusResponse;
import java.util.Map;
import java.util.UUID;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Day082 & Day083:
 * Internal REST bridge between containerized Airflow (`scratchlm_offline_training_v1`)
 * and the PostgreSQL-backed Native Training Worker & Exclusive GPU Supervisor.
 * Authenticated exclusively via `X-Internal-Service-Token`.
 */
@RestController
@RequestMapping("/internal/training-jobs")
public class InternalTrainingJobController {

    public static final String INTERNAL_TOKEN_HEADER = "X-Internal-Service-Token";

    private final TrainingJobAndGpuSupervisorService trainingJobService;

    public InternalTrainingJobController(TrainingJobAndGpuSupervisorService trainingJobService) {
        this.trainingJobService = trainingJobService;
    }

    @PostMapping(consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<TrainingJobStatusResponse> submitTrainingJob(
        @RequestHeader(value = INTERNAL_TOKEN_HEADER, required = false) String internalToken,
        @RequestBody SubmitTrainingJobRequest request
    ) {
        trainingJobService.verifyInternalToken(internalToken);
        TrainingJobStatusResponse response = trainingJobService.submitOrReattachJob(request);
        return ResponseEntity.ok(response);
    }

    @GetMapping(value = "/{jobId}", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<TrainingJobStatusResponse> getTrainingJobStatus(
        @RequestHeader(value = INTERNAL_TOKEN_HEADER, required = false) String internalToken,
        @PathVariable("jobId") UUID jobId
    ) {
        trainingJobService.verifyInternalToken(internalToken);
        return ResponseEntity.ok(trainingJobService.getJobStatus(jobId));
    }

    @PostMapping(value = "/{jobId}/complete", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<TrainingJobStatusResponse> completeTrainingJob(
        @RequestHeader(value = INTERNAL_TOKEN_HEADER, required = false) String internalToken,
        @PathVariable("jobId") UUID jobId,
        @RequestBody(required = false) Map<String, Object> body
    ) {
        trainingJobService.verifyInternalToken(internalToken);
        boolean gatePassed = body == null || !Boolean.FALSE.equals(body.get("validation_gate_passed"));
        return ResponseEntity.ok(trainingJobService.completeRunningJob(jobId, gatePassed));
    }

    @PostMapping(value = "/{jobId}/cancel", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<TrainingJobStatusResponse> cancelTrainingJob(
        @RequestHeader(value = INTERNAL_TOKEN_HEADER, required = false) String internalToken,
        @PathVariable("jobId") UUID jobId,
        @RequestBody(required = false) Map<String, String> body
    ) {
        trainingJobService.verifyInternalToken(internalToken);
        String reason = body != null ? body.get("reason") : null;
        return ResponseEntity.ok(trainingJobService.cancelJob(jobId, reason));
    }
}
