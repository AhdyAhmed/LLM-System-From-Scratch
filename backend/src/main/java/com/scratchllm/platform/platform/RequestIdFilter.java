package com.scratchllm.platform.platform;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.UUID;
import org.slf4j.MDC;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * Assigns or propagates X-Request-Id, X-Correlation-Id, and traceparent on every incoming HTTP request
 * and binds structured correlation metadata into SLF4J MDC without logging sensitive prompts or tokens.
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class RequestIdFilter extends OncePerRequestFilter {

    public static final String REQUEST_ID_HEADER = "X-Request-Id";
    public static final String CORRELATION_ID_HEADER = "X-Correlation-Id";
    public static final String TRACEPARENT_HEADER = "traceparent";
    public static final String REQUEST_ID_ATTR = "scratchllm.request_id";
    public static final String CORRELATION_ID_ATTR = "scratchllm.correlation_id";

    @Override
    protected void doFilterInternal(
        HttpServletRequest request,
        HttpServletResponse response,
        FilterChain filterChain
    ) throws ServletException, IOException {
        String incomingReqId = request.getHeader(REQUEST_ID_HEADER);
        String requestId = (incomingReqId != null && !incomingReqId.isBlank() && incomingReqId.length() <= 64)
            ? incomingReqId.trim()
            : "req-" + UUID.randomUUID();

        String incomingCorrId = request.getHeader(CORRELATION_ID_HEADER);
        String correlationId = (incomingCorrId != null && !incomingCorrId.isBlank() && incomingCorrId.length() <= 64)
            ? incomingCorrId.trim()
            : requestId;

        request.setAttribute(REQUEST_ID_ATTR, requestId);
        request.setAttribute(CORRELATION_ID_ATTR, correlationId);
        response.setHeader(REQUEST_ID_HEADER, requestId);
        response.setHeader(CORRELATION_ID_HEADER, correlationId);

        String traceparent = request.getHeader(TRACEPARENT_HEADER);
        if (traceparent != null && !traceparent.isBlank() && traceparent.length() <= 128) {
            response.setHeader(TRACEPARENT_HEADER, traceparent.trim());
        }

        MDC.put("request_id", requestId);
        MDC.put("correlation_id", correlationId);
        try {
            filterChain.doFilter(request, response);
        } finally {
            MDC.remove("request_id");
            MDC.remove("correlation_id");
        }
    }

    public static String resolveRequestId(HttpServletRequest request) {
        if (request != null) {
            Object attr = request.getAttribute(REQUEST_ID_ATTR);
            if (attr instanceof String s && !s.isBlank()) {
                return s;
            }
        }
        return "req-" + UUID.randomUUID();
    }

    public static String resolveCorrelationId(HttpServletRequest request) {
        if (request != null) {
            Object attr = request.getAttribute(CORRELATION_ID_ATTR);
            if (attr instanceof String s && !s.isBlank()) {
                return s;
            }
        }
        return resolveRequestId(request);
    }
}

