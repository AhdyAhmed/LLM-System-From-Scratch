package com.scratchllm.platform.generation;

import jakarta.validation.Valid;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;

/**
 * Request and response DTOs for idempotent generation run submission, status, and SSE events.
 */
public final class RunDtos {

    private RunDtos() {}

    public record SamplingOptionsDto(
        @NotNull
        @Min(1)
        @Max(64)
        Integer max_new_tokens,

        @NotNull
        @DecimalMin("0.0")
        @DecimalMax("2.0")
        Double temperature,

        @NotNull
        @Min(1)
        @Max(256)
        Integer top_k,

        @NotNull
        @DecimalMin("0.05")
        @DecimalMax("1.0")
        Double top_p,

        Long seed
    ) {}

    public record SubmitRunRequest(
        @NotBlank(message = "Prompt is required")
        @Size(min = 1, max = 16384, message = "Prompt must be between 1 and 16384 characters")
        String prompt,

        @NotBlank(message = "model_alias is required")
        String model_alias,

        @NotNull(message = "sampling options are required")
        @Valid
        SamplingOptionsDto sampling,

        @Size(max = 5, message = "At most 5 document_ids may be attached")
        List<UUID> document_ids
    ) {}

    public record RunAcceptedResponse(
        UUID run_id,
        UUID conversation_id,
        String status,
        String model_revision,
        String events_url,
        String status_url
    ) {}

    public record RunStatusResponse(
        UUID run_id,
        UUID conversation_id,
        String status,
        String model_revision,
        String finish_reason,
        String output_text,
        int reserved_tokens,
        int input_tokens,
        int output_tokens,
        String error_code,
        OffsetDateTime created_at,
        OffsetDateTime completed_at
    ) {}

    public record RunEventRow(
        long sequence,
        String eventType,
        String payloadJson
    ) {}
}
