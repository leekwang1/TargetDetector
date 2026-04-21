#pragma once

#include <cstdarg>
#include <cstdio>

inline void DetectionLogf(const char *tag, const char *fmt, ...)
{
    std::fprintf(stderr, "[%s] ", tag ? tag : "Detection");

    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);

    std::fprintf(stderr, "\n");
}

#define SENDLOGF_TAG(tag, ...) DetectionLogf(tag, __VA_ARGS__)
