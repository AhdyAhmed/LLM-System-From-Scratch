package com.scratchllm.platform.platform;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.UUID;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * Assigns or propagates X-Request-Id on every incoming HTTP request.
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class RequestIdFilter extends OncePerRequestFilter {

    public static final String REQUEST_ID_HEADER = "X-Request-Id";
    public static final String REQUEST_ID_ATTR = "scratchllm.request_id";

    @Override
    protected void doFilterInternal(
        HttpServletRequest request,
        HttpServletResponse response,
        FilterChain filterChain
    ) throws ServletException, IOException {
        String incoming = request.getHeader(REQUEST_ID_HEADER);
        String requestId = (incoming != null && !incoming.isBlank() && incoming.length() <= 64)
            ? incoming.trim()
            : "req-" + UUID.randomUUID();
        request.setAttribute(REQUEST_ID_ATTR, requestId);
        response.setHeader(REQUEST_ID_HEADER, requestId);
        filterChain.doFilter(request, response);
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
}
