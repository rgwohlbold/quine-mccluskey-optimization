#pragma once
#ifndef vct_AARCH64_H
#define vct_AARCH64_H

/* ==================== GNU C and possibly other UNIX compilers ===================== */
#if defined(__GNUC__) || defined(__linux__)
    #define VOLATILE __volatile__
    #define ASM __asm__
#else
    /* if we're neither compiling with gcc or under linux, we can hope
        * the following lines work, they probably won't */
    #define ASM asm
    #define VOLATILE
#endif

#define TIMESTAMP unsigned long long
#define FREQ unsigned long long
#define INT32 unsigned int



/* This is the virtual counter-timer 
* It is a 64-bit register that counts at the frequency of the system's
* physical counter. It is available from ARMv8 onwards.
*
* The frequency of the counter is accessible through the CNTFRQ_EL0 register.
*
* The counter is read by reading the CNTVCT_EL0 register.
*/

#define COUNTER_LO(a) ((a).int32.lo)
#define COUNTER_HI(a) ((a).int32.hi)
#define COUNTER_VAL(a) ((a).int64)

#define COUNTER(a) \
    ((unsigned long long)COUNTER_VAL(a))

#define COUNTER_DIFF(a,b) \
    (COUNTER(a)-COUNTER(b))

#ifdef __aarch64__

    typedef union
    {       TIMESTAMP int64;
            struct {INT32 lo, hi;} int32;
    } vct_counter;

    typedef union
    {       FREQ int64;
            struct {INT32 lo, hi;} int32;
    } vct_freq;

    // virtual counter-timer
    #define CNTVCT(cpu_c) \
        ASM VOLATILE ("mrs %0, cntvct_el0" : "=r" (cpu_c))
    
    // physical counter-timer
    #define CNTPCT(cpu_c) \
        ASM VOLATILE ("mrs %0, cntpct_el0" : "=r" (cpu_c))

    // offset of virtual to physical counter-timer
    #define CNTVOFF(cpu_c) \
        ASM VOLATILE ("mrs %0, cntvoff_el2" : "=r" (cpu_c))

    // frequency of the counter
    #define CNTFRQ(cpu_c) \
        ASM VOLATILE ("mrs %0, cntfrq_el0" : "=r" (cpu_c))
#else // __aarch64__
    #error "Only ARM64 is supported"
#endif // __aarch64__

static inline void init_tsc(void) {
    ;
}
static inline TIMESTAMP start_tsc(void) {
    vct_counter start;
    CNTPCT(start);
    return COUNTER_VAL(start);
}

static inline TIMESTAMP stop_tsc(TIMESTAMP start) {
    vct_counter end;
    CNTPCT(end);
    return COUNTER_VAL(end) - start;
}

static inline TIMESTAMP get_vct_offset(void) {
    vct_counter offset;
    CNTVOFF(offset);
    return COUNTER_VAL(offset);
}

static inline TIMESTAMP get_vct_freq(void) {
    vct_freq freq;
    CNTFRQ(freq);
    return COUNTER_VAL(freq);
}


#endif // vct_AARCH64_H
