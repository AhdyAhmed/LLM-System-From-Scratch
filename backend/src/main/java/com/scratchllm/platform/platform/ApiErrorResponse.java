package com.scratchllm.platform.platform;

import com.fasterxml.jackson.annotation.JsonInclude;
import java.time.OffsetDateTime;
import java.util.Map;

/**
 * Structured machine-readable API error envelope matching contracts/openapi.yaml#/components/schemas/ApiError.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record ApiErrorResponse(
    String error_code,
    String message,
    String request_id,
    OffsetDateTime timestamp,
    Map<String, Object> details
) {}
