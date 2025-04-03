#pragma once

#include <stdbool.h>

#include "implicant.h"

implicant allocate_minterm_array(int num_bits);
bool *allocate_boolean_array(int num_elements);
