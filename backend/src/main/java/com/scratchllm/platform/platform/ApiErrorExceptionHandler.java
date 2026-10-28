package com.scratchllm.platform.platform;

import jakarta.servlet.http.HttpServletRequest;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.data.redis.RedisConnectionFailureException;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingRequestHeaderException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * Central exception translator returning sanitized ApiErrorResponse JSON payloads.
 */
@RestControllerAdvice
public class ApiErrorExceptionHandler {

    @ExceptionHandler(ApiException.class)
    public ResponseEntity<ApiErrorResponse> handleApiException(
        ApiException ex,
        HttpServletRequest request
    ) {
        HttpHeaders headers = new HttpHeaders();
        if (ex.getDetails() != null && ex.getDetails().containsKey("retry_after_seconds")) {
            headers.add(HttpHeaders.RETRY_AFTER, String.valueOf(ex.getDetails().get("retry_after_seconds")));
        }
        ApiErrorResponse body = new ApiErrorResponse(
            ex.getErrorCode(),
            ex.getMessage(),
            RequestIdFilter.resolveRequestId(request),
            OffsetDateTime.now(ZoneOffset.UTC),
            ex.getDetails()
        );
        return new ResponseEntity<>(body, headers, ex.getStatus());
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ApiErrorResponse> handleValidation(
        MethodArgumentNotValidException ex,
        HttpServletRequest request
    ) {
        Map<String, Object> fieldErrors = new LinkedHashMap<>();
        for (FieldError fe : ex.getBindingResult().getFieldErrors()) {
            fieldErrors.put(fe.getField(), fe.getDefaultMessage());
        }
        ApiErrorResponse body = new ApiErrorResponse(
            "VALIDATION_ERROR",
            "Request validation failed.",
            RequestIdFilter.resolveRequestId(request),
            OffsetDateTime.now(ZoneOffset.UTC),
            Map.of("fields", fieldErrors)
        );
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(body);
    }

    @ExceptionHandler({HttpMessageNotReadableException.class, MissingRequestHeaderException.class})
    public ResponseEntity<ApiErrorResponse> handleMalformedRequest(
        Exception ex,
        HttpServletRequest request
    ) {
        ApiErrorResponse body = new ApiErrorResponse(
            "BAD_REQUEST",
            ex instanceof MissingRequestHeaderException mrh
                ? "Missing required header: " + mrh.getHeaderName()
                : "Malformed JSON request body.",
            RequestIdFilter.resolveRequestId(request),
            OffsetDateTime.now(ZoneOffset.UTC),
            null
        );
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(body);
    }

    @ExceptionHandler(RedisConnectionFailureException.class)
    public ResponseEntity<ApiErrorResponse> handleRedisUnavailable(
        RedisConnectionFailureException ex,
        HttpServletRequest request
    ) {
        ApiErrorResponse body = new ApiErrorResponse(
            "REDIS_UNAVAILABLE",
            "Session and rate-limit store is temporarily unavailable; failing closed.",
            RequestIdFilter.resolveRequestId(request),
            OffsetDateTime.now(ZoneOffset.UTC),
            null
        );
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(body);
    }
}
