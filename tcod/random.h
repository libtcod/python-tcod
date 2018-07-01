#ifndef PYTHON_TCOD_RANDOM_H_
#define PYTHON_TCOD_RANDOM_H_

/* define libtcod random functions */

int TCOD_random_get_i(TCOD_random_t mersenne, int min, int max);
double TCOD_random_get_d(TCOD_random_t mersenne, double min, double max);
double TCOD_random_get_gaussian_double(
    TCOD_random_t mersenne, double mean, double std_deviation);
double TCOD_random_get_gaussian_double_range(
    TCOD_random_t mersenne, double min, double max);
double TCOD_random_get_gaussian_double_range_custom(
    TCOD_random_t mersenne, double min, double max, double mean);
double TCOD_random_get_gaussian_double_inv(
    TCOD_random_t mersenne, double mean, double std_deviation);
double TCOD_random_get_gaussian_double_range_inv(
    TCOD_random_t mersenne, double min, double max);
double TCOD_random_get_gaussian_double_range_custom_inv(
    TCOD_random_t mersenne, double min, double max, double mean);

#endif /* PYTHON_TCOD_RANDOM_H_ */
