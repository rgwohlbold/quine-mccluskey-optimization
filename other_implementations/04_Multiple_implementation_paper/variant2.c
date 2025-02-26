#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <stdbool.h>
#include <signal.h>
#include <inttypes.h>
#include <float.h>
#include <ctype.h>
#include <time.h>

/* Global declarations */
#define NMAX 12     /* max number of variables */
#define NTERMS 4096 /* 2 to the NMAX */
#define QTERMS 256  /* NTERMS divided by WORD */
#define MTERMS 4096 /* maximum numbers of minterms */
                    /* allowed */
#define MIMPS 48    /* largest allowable number of */
                    /* prime implicants */
#define LIN 120     /* longest input line size */
#define BIG 32000   /* short integer infinity */
#define WORD 16     /* short integer size */
#define gbit(a, b) (((a) >> (b)) & 1)
int nvars;                   /* number of variables in case */
int nterms = 1;              /* numbers of terms in case */
int nwords;                  /* nterms/WORD + 1 */
int minterm[QTERMS];         /* minterm array (Karnaugh map) */
int noterm[QTERMS];          /* no cares in Karnaugh map */
int impchart[QTERMS][MIMPS]; /* prime implicant chart */
int impcnt[QTERMS];          /* indicates coverage of a minterm */
int impext[QTERMS];          /* indicates multiple coverage of term */
int essprm[MIMPS];           /* marker for essential prime implicants*/
int imps = 0;                /* number of prime implicants */
int priterm[MIMPS];          /* term values for prime implicants */
int pricare[MIMPS];          /* nocare values for prime implicants */
int pptr;                    /* number of actual nterms */
struct list
{
    int term;
    int mom; /* first ancestor */
    int dad; /* second ancestor */
    int nocare;
    int match;
} plist[MTERMS * 2];
char func = 'f';  /* function name */
char vname[NMAX]; /* variable name list */
char *fgets();
main()
{
    int i, j;
    /* input the functions to be reduced */
    getterms();
    if ((nvars > 0) && (nvars <= NMAX))
    {
        /* generate Quine-McCluskey reduction */
        quine();
        /* set up prime implicant chart and determine essential primes */
        i = pchart();

        /* determine minimum function */
        reduction(i);
    }
    getchar();
}
getterms()
{
    char(inline[LIN]);
    register i = 0, j, temp;
    int format;
    int all = 1, none = 0; /* degenerate function checks */
    while (i < QTERMS)
        minterm[i++] = 0;
    /* prompt for format of input */
    fprintf(stdout, "Welcome to the Dallen-Rogers switching function\n");
    fprintf(stdout, "minimization program, Version 1.0, Dec 1, 1981.\n");
    fprintf(stdout, "Enter the number for your preferred input type.\n");
    fprintf(stdout, " 1 - truth table\n");
    fprintf(stdout, " 2 - decimal codes for Karnaugh mapping\n");
    fprintf(stdout, " 3 - logical expression\n");
    fprintf(stdout, "Format number? ");
    fgets(inline, LIN, stdin);
    sscanf(inline, "%d", &format);
    if ((format < 1) || (format > 3))
    {
        fprintf(stdout, "I don't understand your selection. Please try again.\n");
        fprintf(stdout, "format number (1 to 3) ? ");
        fgets(inline, LIN, stdin);
        sscanf(inline, "%d", format);
        if ((format < 1) || (format > 3))
        {
            fprintf(stderr, "invalid input\n");
            nvars = 0;
            return;
        }
    }
    switch (format)
    {
    case 1:
        ttable();
        break;
    case 2:
        kmap();
        break;
    case 3:
        expres();
        break;
    }
    /* check for degenerate cases */
    temp = WORD;

    if (nterms < WORD)
        temp = nterms;
    for (i = 0; i < nwords; i++)
    {
        none |= minterm[i];
        for (j = 0; j < temp; j++)
        {
            all &= gbit(minterm[i] | noterm[i], j);
        }
    }
    if (all != 0)
    {
        fprintf(stdout, "\nfunction = 1\n");
        nvars = 0;
    }
    else if (none == 0)
    {
        fprintf(stdout, "\nfunction = 0\n");
        nvars = 0;
    }
    return;
}
ttable()
{
    char inline[LIN];
    char entry[WORD];
    register i = 0, j, k;
    char *gptr;
    /* instructions for inputting the truth table */
    fprintf(stdout, "Enter each row of your truth table, with input\n");
    fprintf(stdout, "values as 0, 1 or X (not-cares) plus an output\n");
    fprintf(stdout, "value of 0 or 1. Enter an extra RETURN after the\n");
    fprintf(stdout, "last line of the truth table. Delimiters between\n");
    fprintf(stdout, "column entries are not required\n");
    fprintf(stdout, "Row 1: ");
    fgets(inline, LIN, stdin);
    /* Read first line of input to determine the number */
    /* of variables involved */
    nvars = 0;
    i = 0;
    while (inline[i] != '\n')
    {
        if ((inline[i] == '1') || (inline[i] == '0') || (inline[i] == 'x') ||
            (inline[i] == 'X'))
        {
            entry[nvars++] = inline[i];
        }
        i++;
    }
    nvars--;
    /* Determine the maximum numbers of minterms possible */
    /* and the amount of storage required */
    for (i = 0; i < nvars; i++)
    {
        nterms *= 2;
    }
    nwords = ((nterms + (WORD - 1)) / WORD);
    /* set up default variable names */
    for (i = nvars - 1; i >= 0; i--)
    {
#ifdef JMH
        vname[i] = (char)('A' + nvars - i - 1);
#else
        vname[i] = (char)('z' - nvars + i + 1);
#endif
    }
    /* set up defaults as not-cares */
    i = 0;
    while (i < nwords)
    {
        for (j = 0; j < MIMPS; j++)
            impchart[i][j] = 0;
        noterm[i] = ~0;
        minterm[i++] = 0;
    }
    /* zero out unused bits */
    if (nterms < WORD)
    {
        for (j = nterms; j < WORD; j++)
        {
            noterm[0] &= ~(1 << j);
        }
    }
    /* Process each line of truth table and input the next line */
    k = 2;
    while ((gptr != NULL) && (inline[0] != '\n'))
    {
        i = 0;
        j = 0;
        while (inline[i] != '\n')
        {
            if ((inline[i] == '1') || (inline[i] == '0') || (inline[i] == 'x') ||
                (inline[i] == 'X'))
            {
                entry[j++] = inline[i];
            }
            i++;
        }
        ntrmv(0, entry, 0);
        fprintf(stdout, "Row %d: ", k++);
        gptr = fgets(inline, LIN, stdin);
    }
    return;
}
ntrmv(i, entry, trm) register i; /* column number from input table */
char *entry;                     /* pointer to next input term in input string */
register trm;                    /* current minterm configuration */
{

    int num;
    char iterm;
    register j;
    /* have all the input terms been processed ? */
    if (i >= nvars)
    {
        sscanf(entry, "%d", &num);
        minterm[trm / WORD] |= num << (trm % WORD);
        noterm[trm / WORD] &= ~(1 << (trm % WORD));
    }
    else
    {
        sscanf(entry, "%1s", &iterm);
        if ((iterm == '1') || (iterm == '0'))
        {
            trm = (trm << 1) | (iterm - '0');
            ntrmv(i + 1, entry + 1, trm);
        }
        else
        {
            /* any printed symbols other than 0 or 1 are treated */
            /* as not-care terms. */
            trm = trm << 1;
            ntrmv(i + 1, entry + 1, trm);
            trm |= 1;
            ntrmv(i + 1, entry + 1, trm);
        }
    }
    return;
}
kmap()
{
    char inline[LIN];
    char entry[WORD];
    register i = 0, j, temp;
    int num;
    char *gptr;
    fprintf(stdout, "Input the number of variables in your function: ");
    fgets(inline, LIN, stdin);
    sscanf(inline, "%d", &nvars);
    /* Check for valid number of variables */
    if (nvars <= 0)
    {
        nvars = 0;
        return;
    }
    if (nvars > NMAX)
    {
        fprintf(stdout, "Too many variables, %d max\n", NMAX);
        nvars = 0;
        return;
    }
    /* Determine maximum number of minterms and required storage */
    for (i = 0; i < nvars; i++)
    {

        nterms *= 2;
    }
    nwords = ((nterms + (WORD - 1)) / WORD);
    for (i = 0; i < nwords; i++)
    {
        for (j = 0; j < MIMPS; j++)
            impchart[i][j] = 0;
    }
    /* set up default variable names */
    for (i = nvars - 1; i >= 0; i--)
    {
#ifdef JMH
        vname[i] = (char)('A' + nvars - i - 1);
#else
        vname[i] = (char)('z' - nvars + i + 1);
#endif
    }
    fprintf(stdout, "On one line, input each minterm number ");
    fprintf(stdout, "(0 to %d) separated by spaces or tabs:\n", nterms - 1);
    gptr = fgets(inline, LIN, stdin);
    if ((gptr != NULL) && (inline[0] == '\n'))
    {
        nvars = 0;
        return;
    }
    /* Process minterms */
    i = 0;
    while (sscanf(&inline[i], "%s", entry) != EOF)
    {
        j = 0;
        while (entry[j++] != NULL)
            ;
        i += j;
        sscanf(entry, "%d", &num);
        minterm[num / WORD] |= (1 << (num % WORD));
    }
    /* Process not care terms */
    fprintf(stdout, "Input each not-care number (0 to %d), if any:\n", nterms - 1);
    gptr = fgets(inline, LIN, stdin);
    if ((gptr != NULL) && (inline[0] != '\n'))
    {
        i = 0;
        while (sscanf(&inline[i], "%s", entry) != EOF)
        {
            j = 0;
            while (entry[j++] != NULL)
                ;
            i += j;
            sscanf(entry, "%d", &num);
            noterm[num / WORD] |= (1 << (num % WORD));
        }
    }
    return;
}
expres()

{
    char inline[LIN];
    char outline[LIN];
    char tterm[WORD];
    char vdig;
    register i = 0, j = 0, k = 0;
    int flag = 0;
    char *gptr;
    fprintf(stdout, "Input a logical expression, in sum of products form,\n");
    fprintf(stdout, "using single upper or lower case letters as\n");
    fprintf(stdout, "variable names (no subscripts) and a ' following the\n");
    fprintf(stdout, "variable name to indicate a complement. Use a + between\n");
    fprintf(stdout, "terms. Spaces, extraneous symbols, and anything\n");
    fprintf(stdout, "in front of an optional equal sign are ignored.\n? ");
    gptr = fgets(inline, LIN, stdin);
    /* check for bad input */
    nvars = 0;
    if ((gptr == NULL) || (inline[0] == '\n'))
    {
        nvars == 0;
        fprintf(stderr, "no input expression, program terminated\n");
        return;
    }
    /* process expression: eliminate white space, count variables, */
    /* and reverse position of complement sign. */
    while (inline[i] != '\n')
    {
        if (inline[i++] == '=')
            flag = i;
    }
    i = flag;
    while (inline[i] != '\n')
    {
        if (inline[i] == '+')
        {
            outline[j++] = inline[i];
        }
        else if (inline[i] == '\'')
        {
            /* reverse complement sign position */
            outline[j] = outline[j - 1];
            outline[j - 1] = inline[i];
            j++;
        }
        else if (((inline[i] >= 'a') && (inline[i] <= 'z')) ||
                 ((inline[i] >= 'A') && (inline[i] <= 'Z')))
        {
            /* check to see if a new variable */
            outline[j++] = inline[i];
            flag = 0;
            for (k = 0; k < nvars; k++)
            {
                if (vname[k] == inline[i])
                    flag = 1;
            }
            if (flag == 0)
            {
                vname[nvars++] = inline[i];
            }
        }
        i++;
    }
    /* j is length of outline */
    outline[j++] = '+';
    /* Determine maximum number of minterms and required storage */
    for (i = 0; i < nvars; i++)
    {
        nterms *= 2;
    }
    nwords = ((nterms + (WORD - 1)) / WORD);
    /* set up defaults as zero terms */
    i = 0;
    while (i < nwords)
    {
        for (k = 0; k < MIMPS; k++)
            impchart[i][k] = 0;
        noterm[i] = 0;
        minterm[i++] = 0;
    }
    /* check for valid expressions */
    if (nvars == 0)
    {
        fprintf(stderr, "Expression not in readable form\n");
        return;
    }
    /* Convert variables to minterms */
    k = 0;
    while (k < j)
    {
        tterm[nvars] = '1'; /* Sum of products term */
        for (i = 0; i < nvars; i++)
            tterm[i] = 'X';
        while (outline[k] != '+')
        {
            /* build truth table term */
            if (outline[k] == '\'')
            {
                vdig = '0';
                k++;
            }
            else
                vdig = '1';
            for (i = 0; i < nvars; i++)
            {
                if (outline[k] == vname[i])
                {
                    tterm[i] = vdig;
                }
            }
            k++;
        }
        /* set up minterms */
        ntrmv(0, tterm, 0);
        k++;
    }

    return;
}
quine()
{
    register i, j;
    /* initialize prime implicant count */
    for (j = 0; j < nwords; j++)
    {
        impcnt[j] = 0;
        impext[j] = 0;
    }
    for (j = 0; j < MIMPS; j++)
    {
        essprm[j] = 0;
    }
    /* set up pairings list */
    for (i = 0; i < nterms; i++)
    {
        if (gbit((minterm[i / WORD] | noterm[i / WORD]), i % WORD) == 1)
        {
            plist[pptr].nocare = 0;
            plist[pptr].match = 0;
            plist[pptr].mom = 1;
            plist[pptr].dad = 0;
            plist[pptr++].term = i;
            if (pptr >= MTERMS)
            {
                fprintf(stderr, "Too many minterms ( > %d )\n", MTERMS);
                fprintf(stderr, "Process aborted\n");
                i = nterms;
                pptr = 0; /* nullify process */
            }
        }
    }
    /* process pairings */
    pairup(0, pptr);
    return;
}
pairup(first, last) int first, last; /* pointers to first term and last term ( + 1 ) */
                                     /* of candidate terms at one level of Q-M reduc-*/
                                     /* tion. */
{
    int match = 0;       /* indicates a pairing was found */
    int submatch = 0;    /* pairing found on one pass */
    int diff, diffx = 0; /* nocare term variables */
    int fterm, dterm;    /* first term in loop parameters */
    register next;       /* pointer to next available plist location */
    int jstart, second;  /* pointers within the level */
    int i, j2;
    register j, k;

    jstart = first;
    second = first;
    next = last; /* initialize loop controls */
    j = jstart;
    j2 = jstart;
    while (jstart < last - 1)
    {
        while (j2 < (last - 1))
        {
            for (k = second + 1; k < last; k++)
            {
                /* At this point, the full series of Quine-McCluskey 'tests' */
                /* are made to see if a pairing can be made. */
                if ((plist[j].nocare == plist[k].nocare) &&
                    (bitcount(nvars, plist[j].term) == (bitcount(nvars, plist[k].term) - 1)) &&
                    (bitcount(nvars, diff = (plist[k].term ^ plist[j].term)) == 1))
                {
                    if ((diffx == 0) || ((((plist[j].term - fterm) % dterm) == 0) &&
                                         (diff == diffx)))
                    {
                        /* A pairing has been made. Record the pair at the */
                        /* next level. */
                        match = 1;
                        submatch = 1;
                        if (diffx == 0)
                        {
                            dterm = plist[k].term - plist[j].term;
                            fterm = plist[j].term;
                        }
                        plist[j].match = 1;
                        plist[k].match = 1;
                        plist[next].match = 0;
                        plist[next].nocare = plist[j].nocare | diff;
                        plist[next].term = plist[j].term;
                        plist[next].mom = j;
                        plist[next++].dad = k;
                        second = k;
                        diffx = diff;
                        j = ++k;
                    }
                }
            }
            /* A series of tests is made to limit the number of */
            /* possible pairings (without forgetting any), in */
            /* order to accomplish the tabulation recursively. */
            if (submatch == 1)
            {
                second += 2;
                j = second;
                submatch = 0;
            }
            else
            {
                j = ++j;
                j2 = j;
                second = j;
            }
        }
        if (match == 1)
        {
            /* go to the next level of tabulation */
            pairup(last, next);
            j2 = plist[last].mom;
            j = j2;
            second = plist[last].dad;
            next = last;
            match = 0;
            diffx = 0;
        }
        else
        {
            jstart++;
            second = jstart;
            j = jstart;
        }
    }
    /* process the candidate prime implicant */
    primes(first, last);
}
bitcount(len, term) int len; /* length of string to be counted */
register term;               /* string to be counted */
{
    register i;
    register count = 0;
    for (i = 0; i < len; i++)
        count += (term >> i) & 1;
    return (count);
}
primes(first, last) int first, last;
{
    register i, j;
    int flag;
    int rep;
    int match;
    /* output prime implicants */
    for (j = first; j < last; j++)
    {
        if (plist[j].match == 0)
        {
            flag = 0;
            for (i = 0; i < imps; i++)
            {
                /* test to see if candidate is a subset of a larger prime imp */

                if (bitcount(nvars, plist[j].nocare) <= bitcount(nvars, pricare[i]))
                {
                    if (((plist[j].nocare | pricare[i]) == (pricare[i])) &&
                        (((~pricare[i]) & priterm[i]) == ((~pricare[i]) & plist[j].term)) &&
                        ((plist[j].term | priterm[i] | pricare[i]) ==
                         (priterm[i] | pricare[i])))
                    {
                        flag = 1;
                    }
                    /* test to see if candidate will replace a smaller subset */
                }
                else if (bitcount(nvars, plist[j].nocare) > bitcount(nvars, pricare[i]))
                {
                    if (((pricare[i] | plist[j].nocare) == (plist[j].nocare)) &&
                        (((~plist[j].nocare) & plist[j].term) ==
                         ((~plist[j].nocare) & priterm[i])) &&
                        ((priterm[i] | plist[j].term | plist[j].nocare) ==
                         (plist[j].term | plist[j].nocare)))
                    {
                        flag = 2;
                        rep = i;
                    }
                }
            }
            /* Add a prime implicant to list--no complications */
            if (flag == 0)
            {
                primepath(j, imps);
                priterm[imps] = plist[j].term;
                pricare[imps] = plist[j].nocare;
                imps++;
                if (imps >= MIMPS)
                {
                    fprintf(stderr, "Warning, overflow of prime implicant chart\n");
                    imps--; /* for protection */
                }
                /* Preform the replacement of a prime implicat with a larger one */
            }
            else if (flag == 2)
            {
                primepath(j, rep);
                priterm[rep] = plist[j].term;
                pricare[rep] = plist[j].nocare;
            }
        }
    }
    return;
}
primepath(j, imp) register j; /* start node in Quine-McCluskey */
register imp;                 /* entry in implicant table */
{
    if (j < pptr)
    {

        /* arrival back at original terms */
        impchart[plist[j].term / WORD][imp] |= (1 << (plist[j].term % WORD));
    }
    else
    {
        primepath(plist[j].mom, imp);
        primepath(plist[j].dad, imp);
    }
    return;
}
pchart()
{
    register i, j, k;
    int uncov;
    int temp;
    char echar;
    /* determine coverage of minterms */
    for (i = 0; i < nwords; i++)
    {
        for (j = 0; j < imps; j++)
        {
            impcnt[i] |= impchart[i][j];
        }
    }
    /* determine multiple coverage of minterms */
    for (i = 0; i < nterms; i++)
    {
        temp = 0;
        for (j = 0; j < imps; j++)
        {
            temp += (gbit(impchart[i / WORD][j], i % WORD));
        }
        if (temp >= 2)
            impext[i / WORD] |= (1 << (i % WORD));
    }
    /* exclude not-care terms from consideration */
    for (i = 0; i < nwords; i++)
    {
        /* eliminate not-care cases */
        impcnt[i] &= ~noterm[i];
        impext[i] &= ~noterm[i];
        /* check for prime implicants */
        for (j = 0; j < WORD; j++)
        {
            if ((gbit(impcnt[i], j) == 1) && (gbit(impext[i], j) == 0))
            {
                k = 0;
                while (gbit(impchart[i][k], j) == 0)
                    k++;
                essprm[k] = 1;
            }
        }
    }
    /* Determine coverage of essential prime implicants and */
    /* print out prime implicants */
    fprintf(stdout, "\nPrime implicants ( * indicates essential prime implicant )");
    for (i = 0; i < imps; i++)
    {

        if (essprm[i] == 1)
        {
            echar = '*';
            for (j = 0; j < nwords; j++)
            {
                impcnt[j] &= ~impchart[j][i];
            }
        }
        else
            echar = ' ';
        fprintf(stdout, "\n%c ", echar);
        for (j = 0; j < nvars; j++)
        {
            if (((pricare[i] >> (nvars - 1 - j)) & 1) == 0)
            {
                fprintf(stdout, "%c", vname[j]);
                if (((priterm[i] >> (nvars - 1 - j)) & 1) == 0)
                {
                    fprintf(stdout, "'");
                }
            }
        }
        fprintf(stdout, "\t: ");
        for (j = 0; j < nterms; j++)
        {
            if (gbit(impchart[j / WORD][i], j % WORD) == 1)
                fprintf(stdout, "%d,", j);
        }
    }
    uncov = 0;
    for (i = 0; i < nwords; i++)
    {
        uncov += bitcount(WORD, impcnt[i]);
    }
    return (uncov);
}
reduction(uncov) int uncov;
{
    register i, j;
    int nonemps;          /* number of non-essential terms */
    int terms, lits;      /* minimization factors */
    int nons[MIMPS];      /* index into impchart of non-essential impl.*/
    int termlist[QTERMS]; /* temporary location of covered term count */
    int fail;             /* new candidate flag */
    long limit, li;       /* power set bits */
    char oper;            /* sum of products separator */
    /* current best coverage candidate */
    struct current
    {
        int terms;
        int lits;
        int list[MIMPS];
    } curbest;
    if (uncov == 0)
    {
        fprintf(stdout, "\n\nminimal expression is unique\n");
    }
    else
    {
        fprintf(stdout, "\n\nno unique minimal expression\n");
        /* set up non-essential implicant list */
        j = 0;
        for (i = 0; i < imps; i++)
            if (essprm[i] == 0)
                nons[j++] = i;
        nonemps = j;
        /* insure no overflow of cyclical prime implicant array */
        if (nonemps > 2 * WORD)
        {
            fprintf(stderr, "Warning! Only %d prime implicants can be", 2 * WORD);
            fprintf(stderr, "considered for coverage\n of all terms (in addition");
            fprintf(stderr, "to essential primes). %d implicants not checked\n",
                    nonemps - (2 * WORD));
            nonemps = 2 * WORD;
        }
        if (nonemps > WORD)
        {
            fprintf(stdout, "Warning! Large number of cyclical prime implicants\n");
            fprintf(stdout, "Computation will take awhile\n");
        }
        /* candidate coverage is determined by generation of the power set */
        /* calculate power set */
        limit = 1;
        for (i = 0; i < nonemps; i++)
            limit *= 2;
        /* set up current best expression list */
        curbest.terms = BIG;
        curbest.lits = BIG;
        curbest.list[0] = -1;
        /* try each case */
        for (li = 1L; li < limit; li++)
        {
            terms = bitcount(2 * WORD, li);
            if (terms <= curbest.terms)
            {
                /* reset count */
                lits = 0;
                /* reset uncovered term list */
                for (i = 0; i < nwords; i++)
                    termlist[i] = impcnt[i];
                for (i = 0; i < nonemps; i++)
                {
                    if (((li >> i) & 1L) == 1L)
                    {
                        for (j = 0; j < nterms; j++)
                        {
                            if (gbit(impchart[j / WORD][nons[i]], j % WORD) == 1)
                            {
                                termlist[j / WORD] &= ~(1 << (j % WORD));
                            }
                        }
                    }
                    lits += (nvars - bitcount(nvars, pricare[nons[i]]));
                }

                fail = 0;
                for (i = 0; i < nwords; i++)
                {
                    if (termlist[i] != 0)
                        fail = 1;
                }
                if ((fail == 0) && ((terms < curbest.terms) || (lits < curbest.lits)))
                {
                    /* we have a new candidate */
                    curbest.terms = terms;
                    curbest.lits = lits;
                    j = 0;
                    for (i = 0; i < nonemps; i++)
                    {
                        if (((li >> i) & 1L) == 1L)
                        {
                            curbest.list[j++] = nons[i];
                        }
                    }
                    curbest.list[j] = -1;
                }
            }
        }
        j = 0;
        while (curbest.list[j] >= 0)
        {
            essprm[curbest.list[j]] = 1;
            j++;
        }
    }
    /* print out minimal expression */
    fprintf(stdout, "\nminimal expression:\n\n %c(", func);
    for (i = 0; i < nvars; i++)
    {
        fprintf(stdout, "%c", vname[i]);
        if (i < nvars - 1)
            fprintf(stdout, ",");
    }
    fprintf(stdout, ")");
    oper = '=';
    for (i = 0; i < imps; i++)
    {
        if (essprm[i] == 1)
        {
            fprintf(stdout, " %c ", oper);
            for (j = 0; j < nvars; j++)
            {
                if (((pricare[i] >> (nvars - 1 - j)) & 1) == 0)
                {
                    fprintf(stdout, "%c", vname[j]);
                    if (((priterm[i] >> (nvars - 1 - j)) & 1) == 0)
                    {
                        fprintf(stdout, "'");
                    }
                }
            }
            oper = '+';
        }
    }
    fprintf(stdout, "\n\n");
    return;
}