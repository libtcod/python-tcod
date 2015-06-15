#define _CFFI_
#include <Python.h>
#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>

/* See doc/misc/parse_c_type.rst in the source of CFFI for more information */

typedef void *_cffi_opcode_t;

#define _CFFI_OP(opcode, arg)   (_cffi_opcode_t)(opcode | (((uintptr_t)(arg)) << 8))
#define _CFFI_GETOP(cffi_opcode)    ((unsigned char)(uintptr_t)cffi_opcode)
#define _CFFI_GETARG(cffi_opcode)   (((uintptr_t)cffi_opcode) >> 8)

#define _CFFI_OP_PRIMITIVE       1
#define _CFFI_OP_POINTER         3
#define _CFFI_OP_ARRAY           5
#define _CFFI_OP_OPEN_ARRAY      7
#define _CFFI_OP_STRUCT_UNION    9
#define _CFFI_OP_ENUM           11
#define _CFFI_OP_FUNCTION       13
#define _CFFI_OP_FUNCTION_END   15
#define _CFFI_OP_NOOP           17
#define _CFFI_OP_BITFIELD       19
#define _CFFI_OP_TYPENAME       21
#define _CFFI_OP_CPYTHON_BLTN_V 23   // varargs
#define _CFFI_OP_CPYTHON_BLTN_N 25   // noargs
#define _CFFI_OP_CPYTHON_BLTN_O 27   // O  (i.e. a single arg)
#define _CFFI_OP_CONSTANT       29
#define _CFFI_OP_CONSTANT_INT   31
#define _CFFI_OP_GLOBAL_VAR     33
#define _CFFI_OP_DLOPEN_FUNC    35
#define _CFFI_OP_DLOPEN_CONST   37

#define _CFFI_PRIM_VOID          0
#define _CFFI_PRIM_BOOL          1
#define _CFFI_PRIM_CHAR          2
#define _CFFI_PRIM_SCHAR         3
#define _CFFI_PRIM_UCHAR         4
#define _CFFI_PRIM_SHORT         5
#define _CFFI_PRIM_USHORT        6
#define _CFFI_PRIM_INT           7
#define _CFFI_PRIM_UINT          8
#define _CFFI_PRIM_LONG          9
#define _CFFI_PRIM_ULONG        10
#define _CFFI_PRIM_LONGLONG     11
#define _CFFI_PRIM_ULONGLONG    12
#define _CFFI_PRIM_FLOAT        13
#define _CFFI_PRIM_DOUBLE       14
#define _CFFI_PRIM_LONGDOUBLE   15

#define _CFFI_PRIM_WCHAR        16
#define _CFFI_PRIM_INT8         17
#define _CFFI_PRIM_UINT8        18
#define _CFFI_PRIM_INT16        19
#define _CFFI_PRIM_UINT16       20
#define _CFFI_PRIM_INT32        21
#define _CFFI_PRIM_UINT32       22
#define _CFFI_PRIM_INT64        23
#define _CFFI_PRIM_UINT64       24
#define _CFFI_PRIM_INTPTR       25
#define _CFFI_PRIM_UINTPTR      26
#define _CFFI_PRIM_PTRDIFF      27
#define _CFFI_PRIM_SIZE         28
#define _CFFI_PRIM_SSIZE        29
#define _CFFI_PRIM_INT_LEAST8   30
#define _CFFI_PRIM_UINT_LEAST8  31
#define _CFFI_PRIM_INT_LEAST16  32
#define _CFFI_PRIM_UINT_LEAST16 33
#define _CFFI_PRIM_INT_LEAST32  34
#define _CFFI_PRIM_UINT_LEAST32 35
#define _CFFI_PRIM_INT_LEAST64  36
#define _CFFI_PRIM_UINT_LEAST64 37
#define _CFFI_PRIM_INT_FAST8    38
#define _CFFI_PRIM_UINT_FAST8   39
#define _CFFI_PRIM_INT_FAST16   40
#define _CFFI_PRIM_UINT_FAST16  41
#define _CFFI_PRIM_INT_FAST32   42
#define _CFFI_PRIM_UINT_FAST32  43
#define _CFFI_PRIM_INT_FAST64   44
#define _CFFI_PRIM_UINT_FAST64  45
#define _CFFI_PRIM_INTMAX       46
#define _CFFI_PRIM_UINTMAX      47

#define _CFFI__NUM_PRIM         48
#define _CFFI__UNKNOWN_PRIM    (-1)


struct _cffi_global_s {
    const char *name;
    void *address;
    _cffi_opcode_t type_op;
    void *size_or_direct_fn;  // OP_GLOBAL_VAR: size, or 0 if unknown
                              // OP_CPYTHON_BLTN_*: addr of direct function
};

struct _cffi_getconst_s {
    unsigned long long value;
    const struct _cffi_type_context_s *ctx;
    int gindex;
};

struct _cffi_struct_union_s {
    const char *name;
    int type_index;          // -> _cffi_types, on a OP_STRUCT_UNION
    int flags;               // _CFFI_F_* flags below
    size_t size;
    int alignment;
    int first_field_index;   // -> _cffi_fields array
    int num_fields;
};
#define _CFFI_F_UNION         0x01   // is a union, not a struct
#define _CFFI_F_CHECK_FIELDS  0x02   // complain if fields are not in the
                                     // "standard layout" or if some are missing
#define _CFFI_F_PACKED        0x04   // for CHECK_FIELDS, assume a packed struct
#define _CFFI_F_EXTERNAL      0x08   // in some other ffi.include()
#define _CFFI_F_OPAQUE        0x10   // opaque

struct _cffi_field_s {
    const char *name;
    size_t field_offset;
    size_t field_size;
    _cffi_opcode_t field_type_op;
};

struct _cffi_enum_s {
    const char *name;
    int type_index;          // -> _cffi_types, on a OP_ENUM
    int type_prim;           // _CFFI_PRIM_xxx
    const char *enumerators; // comma-delimited string
};

struct _cffi_typename_s {
    const char *name;
    int type_index;   /* if opaque, points to a possibly artificial
                         OP_STRUCT which is itself opaque */
};

struct _cffi_type_context_s {
    _cffi_opcode_t *types;
    const struct _cffi_global_s *globals;
    const struct _cffi_field_s *fields;
    const struct _cffi_struct_union_s *struct_unions;
    const struct _cffi_enum_s *enums;
    const struct _cffi_typename_s *typenames;
    int num_globals;
    int num_struct_unions;
    int num_enums;
    int num_typenames;
    const char *const *includes;
    int num_types;
    int flags;      /* future extension */
};

struct _cffi_parse_info_s {
    const struct _cffi_type_context_s *ctx;
    _cffi_opcode_t *output;
    unsigned int output_size;
    size_t error_location;
    const char *error_message;
};

#ifdef _CFFI_INTERNAL
static int parse_c_type(struct _cffi_parse_info_s *info, const char *input);
static int search_in_globals(const struct _cffi_type_context_s *ctx,
                             const char *search, size_t search_len);
static int search_in_struct_unions(const struct _cffi_type_context_s *ctx,
                                   const char *search, size_t search_len);
#endif

/* this block of #ifs should be kept exactly identical between
   c/_cffi_backend.c, cffi/vengine_cpy.py, cffi/vengine_gen.py
   and cffi/_cffi_include.h */
#if defined(_MSC_VER)
# include <malloc.h>   /* for alloca() */
# if _MSC_VER < 1600   /* MSVC < 2010 */
   typedef __int8 int8_t;
   typedef __int16 int16_t;
   typedef __int32 int32_t;
   typedef __int64 int64_t;
   typedef unsigned __int8 uint8_t;
   typedef unsigned __int16 uint16_t;
   typedef unsigned __int32 uint32_t;
   typedef unsigned __int64 uint64_t;
   typedef __int8 int_least8_t;
   typedef __int16 int_least16_t;
   typedef __int32 int_least32_t;
   typedef __int64 int_least64_t;
   typedef unsigned __int8 uint_least8_t;
   typedef unsigned __int16 uint_least16_t;
   typedef unsigned __int32 uint_least32_t;
   typedef unsigned __int64 uint_least64_t;
   typedef __int8 int_fast8_t;
   typedef __int16 int_fast16_t;
   typedef __int32 int_fast32_t;
   typedef __int64 int_fast64_t;
   typedef unsigned __int8 uint_fast8_t;
   typedef unsigned __int16 uint_fast16_t;
   typedef unsigned __int32 uint_fast32_t;
   typedef unsigned __int64 uint_fast64_t;
   typedef __int64 intmax_t;
   typedef unsigned __int64 uintmax_t;
# else
#  include <stdint.h>
# endif
# if _MSC_VER < 1800   /* MSVC < 2013 */
   typedef unsigned char _Bool;
# endif
#else
# include <stdint.h>
# if (defined (__SVR4) && defined (__sun)) || defined(_AIX)
#  include <alloca.h>
# endif
#endif

#ifdef __GNUC__
# define _CFFI_UNUSED_FN  __attribute__((unused))
#else
# define _CFFI_UNUSED_FN  /* nothing */
#endif

/**********  CPython-specific section  **********/
#ifndef PYPY_VERSION


#if PY_MAJOR_VERSION >= 3
# define PyInt_FromLong PyLong_FromLong
#endif

#define _cffi_from_c_double PyFloat_FromDouble
#define _cffi_from_c_float PyFloat_FromDouble
#define _cffi_from_c_long PyInt_FromLong
#define _cffi_from_c_ulong PyLong_FromUnsignedLong
#define _cffi_from_c_longlong PyLong_FromLongLong
#define _cffi_from_c_ulonglong PyLong_FromUnsignedLongLong

#define _cffi_to_c_double PyFloat_AsDouble
#define _cffi_to_c_float PyFloat_AsDouble

#define _cffi_from_c_int(x, type)                                        \
    (((type)-1) > 0 ? /* unsigned */                                     \
        (sizeof(type) < sizeof(long) ?                                   \
            PyInt_FromLong((long)x) :                                    \
         sizeof(type) == sizeof(long) ?                                  \
            PyLong_FromUnsignedLong((unsigned long)x) :                  \
            PyLong_FromUnsignedLongLong((unsigned long long)x)) :        \
        (sizeof(type) <= sizeof(long) ?                                  \
            PyInt_FromLong((long)x) :                                    \
            PyLong_FromLongLong((long long)x)))

#define _cffi_to_c_int(o, type)                                          \
    ((type)(                                                             \
     sizeof(type) == 1 ? (((type)-1) > 0 ? (type)_cffi_to_c_u8(o)        \
                                         : (type)_cffi_to_c_i8(o)) :     \
     sizeof(type) == 2 ? (((type)-1) > 0 ? (type)_cffi_to_c_u16(o)       \
                                         : (type)_cffi_to_c_i16(o)) :    \
     sizeof(type) == 4 ? (((type)-1) > 0 ? (type)_cffi_to_c_u32(o)       \
                                         : (type)_cffi_to_c_i32(o)) :    \
     sizeof(type) == 8 ? (((type)-1) > 0 ? (type)_cffi_to_c_u64(o)       \
                                         : (type)_cffi_to_c_i64(o)) :    \
     (Py_FatalError("unsupported size for type " #type), (type)0)))

#define _cffi_to_c_i8                                                    \
                 ((int(*)(PyObject *))_cffi_exports[1])
#define _cffi_to_c_u8                                                    \
                 ((int(*)(PyObject *))_cffi_exports[2])
#define _cffi_to_c_i16                                                   \
                 ((int(*)(PyObject *))_cffi_exports[3])
#define _cffi_to_c_u16                                                   \
                 ((int(*)(PyObject *))_cffi_exports[4])
#define _cffi_to_c_i32                                                   \
                 ((int(*)(PyObject *))_cffi_exports[5])
#define _cffi_to_c_u32                                                   \
                 ((unsigned int(*)(PyObject *))_cffi_exports[6])
#define _cffi_to_c_i64                                                   \
                 ((long long(*)(PyObject *))_cffi_exports[7])
#define _cffi_to_c_u64                                                   \
                 ((unsigned long long(*)(PyObject *))_cffi_exports[8])
#define _cffi_to_c_char                                                  \
                 ((int(*)(PyObject *))_cffi_exports[9])
#define _cffi_from_c_pointer                                             \
    ((PyObject *(*)(char *, CTypeDescrObject *))_cffi_exports[10])
#define _cffi_to_c_pointer                                               \
    ((char *(*)(PyObject *, CTypeDescrObject *))_cffi_exports[11])
#define _cffi_get_struct_layout                                          \
    not used any more
#define _cffi_restore_errno                                              \
    ((void(*)(void))_cffi_exports[13])
#define _cffi_save_errno                                                 \
    ((void(*)(void))_cffi_exports[14])
#define _cffi_from_c_char                                                \
    ((PyObject *(*)(char))_cffi_exports[15])
#define _cffi_from_c_deref                                               \
    ((PyObject *(*)(char *, CTypeDescrObject *))_cffi_exports[16])
#define _cffi_to_c                                                       \
    ((int(*)(char *, CTypeDescrObject *, PyObject *))_cffi_exports[17])
#define _cffi_from_c_struct                                              \
    ((PyObject *(*)(char *, CTypeDescrObject *))_cffi_exports[18])
#define _cffi_to_c_wchar_t                                               \
    ((wchar_t(*)(PyObject *))_cffi_exports[19])
#define _cffi_from_c_wchar_t                                             \
    ((PyObject *(*)(wchar_t))_cffi_exports[20])
#define _cffi_to_c_long_double                                           \
    ((long double(*)(PyObject *))_cffi_exports[21])
#define _cffi_to_c__Bool                                                 \
    ((_Bool(*)(PyObject *))_cffi_exports[22])
#define _cffi_prepare_pointer_call_argument                              \
    ((Py_ssize_t(*)(CTypeDescrObject *, PyObject *, char **))_cffi_exports[23])
#define _cffi_convert_array_from_object                                  \
    ((int(*)(char *, CTypeDescrObject *, PyObject *))_cffi_exports[24])
#define _CFFI_NUM_EXPORTS 25

typedef struct _ctypedescr CTypeDescrObject;

static void *_cffi_exports[_CFFI_NUM_EXPORTS];

#define _cffi_type(index)   (                           \
    assert((((uintptr_t)_cffi_types[index]) & 1) == 0), \
    (CTypeDescrObject *)_cffi_types[index])

static PyObject *_cffi_init(const char *module_name, Py_ssize_t version,
                            const struct _cffi_type_context_s *ctx)
{
    PyObject *module, *o_arg, *new_module;
    void *raw[] = {
        (void *)module_name,
        (void *)version,
        (void *)_cffi_exports,
        (void *)ctx,
    };

    module = PyImport_ImportModule("_cffi_backend");
    if (module == NULL)
        goto failure;

    o_arg = PyLong_FromVoidPtr((void *)raw);
    if (o_arg == NULL)
        goto failure;

    new_module = PyObject_CallMethod(
        module, (char *)"_init_cffi_1_0_external_module", (char *)"O", o_arg);

    Py_DECREF(o_arg);
    Py_DECREF(module);
    return new_module;

  failure:
    Py_XDECREF(module);
    return NULL;
}

_CFFI_UNUSED_FN
static PyObject **_cffi_unpack_args(PyObject *args_tuple, Py_ssize_t expected,
                                    const char *fnname)
{
    if (PyTuple_GET_SIZE(args_tuple) != expected) {
        PyErr_Format(PyExc_TypeError,
                     "%.150s() takes exactly %zd arguments (%zd given)",
                     fnname, expected, PyTuple_GET_SIZE(args_tuple));
        return NULL;
    }
    return &PyTuple_GET_ITEM(args_tuple, 0);   /* pointer to the first item,
                                                  the others follow */
}

#endif
/**********  end CPython-specific section  **********/


#define _cffi_array_len(array)   (sizeof(array) / sizeof((array)[0]))

#define _cffi_prim_int(size, sign)                                      \
    ((size) == 1 ? ((sign) ? _CFFI_PRIM_INT8  : _CFFI_PRIM_UINT8)  :    \
     (size) == 2 ? ((sign) ? _CFFI_PRIM_INT16 : _CFFI_PRIM_UINT16) :    \
     (size) == 4 ? ((sign) ? _CFFI_PRIM_INT32 : _CFFI_PRIM_UINT32) :    \
     (size) == 8 ? ((sign) ? _CFFI_PRIM_INT64 : _CFFI_PRIM_UINT64) :    \
     _CFFI__UNKNOWN_PRIM)

#define _cffi_check_int(got, got_nonpos, expected)      \
    ((got_nonpos) == (expected <= 0) &&                 \
     (got) == (unsigned long long)expected)

#ifdef __cplusplus
}
#endif

/************************************************************/


#include <libtcod.h>

static void set_char(TCOD_console_t console, int x, int y,
                     int ch, int fg, int bg){
    TCOD_color_t color;
    if(ch != -1){
        TCOD_console_set_char(console, x, y, ch);
    }
    if(fg != -1){
        color.r = fg >> 16 & 0xff;
        color.g = fg >> 8 & 0xff;
        color.b = fg & 0xff;
        TCOD_console_set_char_foreground(console, x, y, color);
    }
    if(bg != -1){
        color.r = bg >> 16 & 0xff;
        color.g = bg >> 8 & 0xff;
        color.b = bg & 0xff;
        TCOD_console_set_char_background(console, x, y, color, 1);
    }
    
}



/************************************************************/

static void *_cffi_types[] = {
/*  0 */ _CFFI_OP(_CFFI_OP_FUNCTION, 63), // TCOD_alignment_t()(void *)
/*  1 */ _CFFI_OP(_CFFI_OP_POINTER, 400), // void *
/*  2 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/*  3 */ _CFFI_OP(_CFFI_OP_FUNCTION, 62), // TCOD_bkgnd_flag_t()(void *)
/*  4 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/*  5 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/*  6 */ _CFFI_OP(_CFFI_OP_FUNCTION, 138), // TCOD_color_t()(void *)
/*  7 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/*  8 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/*  9 */ _CFFI_OP(_CFFI_OP_FUNCTION, 138), // TCOD_color_t()(void *, float, float, float, float)
/* 10 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 11 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 13), // float
/* 12 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 13),
/* 13 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 13),
/* 14 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 13),
/* 15 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 16 */ _CFFI_OP(_CFFI_OP_FUNCTION, 138), // TCOD_color_t()(void *, int, int)
/* 17 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 18 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7), // int
/* 19 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 20 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 21 */ _CFFI_OP(_CFFI_OP_FUNCTION, 138), // TCOD_color_t()(void)
/* 22 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 23 */ _CFFI_OP(_CFFI_OP_FUNCTION, 379), // TCOD_event_t()(int, TCOD_key_t *, TCOD_mouse_t *)
/* 24 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 25 */ _CFFI_OP(_CFFI_OP_POINTER, 382), // TCOD_key_t *
/* 26 */ _CFFI_OP(_CFFI_OP_POINTER, 383), // TCOD_mouse_t *
/* 27 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 28 */ _CFFI_OP(_CFFI_OP_FUNCTION, 379), // TCOD_event_t()(int, TCOD_key_t *, TCOD_mouse_t *, unsigned char)
/* 29 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 30 */ _CFFI_OP(_CFFI_OP_NOOP, 25),
/* 31 */ _CFFI_OP(_CFFI_OP_NOOP, 26),
/* 32 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 4), // unsigned char
/* 33 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 34 */ _CFFI_OP(_CFFI_OP_FUNCTION, 382), // TCOD_key_t()(int)
/* 35 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 36 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 37 */ _CFFI_OP(_CFFI_OP_FUNCTION, 382), // TCOD_key_t()(unsigned char)
/* 38 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 4),
/* 39 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 40 */ _CFFI_OP(_CFFI_OP_FUNCTION, 383), // TCOD_mouse_t()(void)
/* 41 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 42 */ _CFFI_OP(_CFFI_OP_FUNCTION, 142), // TCOD_renderer_t()(void)
/* 43 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 44 */ _CFFI_OP(_CFFI_OP_FUNCTION, 384), // char *()(void)
/* 45 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 46 */ _CFFI_OP(_CFFI_OP_FUNCTION, 11), // float()(void)
/* 47 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 48 */ _CFFI_OP(_CFFI_OP_FUNCTION, 18), // int()(void *)
/* 49 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 50 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 51 */ _CFFI_OP(_CFFI_OP_FUNCTION, 18), // int()(void *, int, int)
/* 52 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 53 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 54 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 55 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 56 */ _CFFI_OP(_CFFI_OP_FUNCTION, 18), // int()(void *, int, int, int, int, TCOD_bkgnd_flag_t, TCOD_alignment_t, char const *, ...)
/* 57 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 58 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 59 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 60 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 61 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 62 */ _CFFI_OP(_CFFI_OP_ENUM, 1), // TCOD_bkgnd_flag_t
/* 63 */ _CFFI_OP(_CFFI_OP_ENUM, 0), // TCOD_alignment_t
/* 64 */ _CFFI_OP(_CFFI_OP_POINTER, 385), // char const *
/* 65 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 1),
/* 66 */ _CFFI_OP(_CFFI_OP_FUNCTION, 18), // int()(void *, int, int, int, int, TCOD_bkgnd_flag_t, TCOD_alignment_t, wchar_t const *, ...)
/* 67 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 68 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 69 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 70 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 71 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 72 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 73 */ _CFFI_OP(_CFFI_OP_NOOP, 63),
/* 74 */ _CFFI_OP(_CFFI_OP_POINTER, 401), // wchar_t const *
/* 75 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 1),
/* 76 */ _CFFI_OP(_CFFI_OP_FUNCTION, 18), // int()(void *, int, int, int, int, char const *, ...)
/* 77 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 78 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 79 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 80 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 81 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 82 */ _CFFI_OP(_CFFI_OP_NOOP, 64),
/* 83 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 1),
/* 84 */ _CFFI_OP(_CFFI_OP_FUNCTION, 18), // int()(void *, int, int, int, int, wchar_t const *, ...)
/* 85 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 86 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 87 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 88 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 89 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 90 */ _CFFI_OP(_CFFI_OP_NOOP, 74),
/* 91 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 1),
/* 92 */ _CFFI_OP(_CFFI_OP_FUNCTION, 18), // int()(void)
/* 93 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 94 */ _CFFI_OP(_CFFI_OP_FUNCTION, 32), // unsigned char()(TCOD_keycode_t)
/* 95 */ _CFFI_OP(_CFFI_OP_ENUM, 6), // TCOD_keycode_t
/* 96 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 97 */ _CFFI_OP(_CFFI_OP_FUNCTION, 32), // unsigned char()(int, int, unsigned char)
/* 98 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 99 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 100 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 4),
/* 101 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 102 */ _CFFI_OP(_CFFI_OP_FUNCTION, 32), // unsigned char()(void *, char const *)
/* 103 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 104 */ _CFFI_OP(_CFFI_OP_NOOP, 64),
/* 105 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 106 */ _CFFI_OP(_CFFI_OP_FUNCTION, 32), // unsigned char()(void *, int, int)
/* 107 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 108 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 109 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 110 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 111 */ _CFFI_OP(_CFFI_OP_FUNCTION, 32), // unsigned char()(void)
/* 112 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 113 */ _CFFI_OP(_CFFI_OP_FUNCTION, 1), // void *()(char const *)
/* 114 */ _CFFI_OP(_CFFI_OP_NOOP, 64),
/* 115 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 116 */ _CFFI_OP(_CFFI_OP_FUNCTION, 1), // void *()(int(*)(void *), void *)
/* 117 */ _CFFI_OP(_CFFI_OP_POINTER, 48), // int(*)(void *)
/* 118 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 119 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 120 */ _CFFI_OP(_CFFI_OP_FUNCTION, 1), // void *()(int)
/* 121 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 122 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 123 */ _CFFI_OP(_CFFI_OP_FUNCTION, 1), // void *()(int, int)
/* 124 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 125 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 126 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 127 */ _CFFI_OP(_CFFI_OP_FUNCTION, 1), // void *()(void *)
/* 128 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 129 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 130 */ _CFFI_OP(_CFFI_OP_FUNCTION, 1), // void *()(void *, char const *)
/* 131 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 132 */ _CFFI_OP(_CFFI_OP_NOOP, 64),
/* 133 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 134 */ _CFFI_OP(_CFFI_OP_FUNCTION, 1), // void *()(void)
/* 135 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 136 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(TCOD_colctrl_t, TCOD_color_t, TCOD_color_t)
/* 137 */ _CFFI_OP(_CFFI_OP_ENUM, 2), // TCOD_colctrl_t
/* 138 */ _CFFI_OP(_CFFI_OP_STRUCT_UNION, 0), // TCOD_color_t
/* 139 */ _CFFI_OP(_CFFI_OP_NOOP, 138),
/* 140 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 141 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(TCOD_renderer_t)
/* 142 */ _CFFI_OP(_CFFI_OP_ENUM, 7), // TCOD_renderer_t
/* 143 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 144 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(char const *)
/* 145 */ _CFFI_OP(_CFFI_OP_NOOP, 64),
/* 146 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 147 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(char const *, int, int)
/* 148 */ _CFFI_OP(_CFFI_OP_NOOP, 64),
/* 149 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 150 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 151 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 152 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(char const *, int, int, int)
/* 153 */ _CFFI_OP(_CFFI_OP_NOOP, 64),
/* 154 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 155 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 156 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 157 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 158 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(int *, int *)
/* 159 */ _CFFI_OP(_CFFI_OP_POINTER, 18), // int *
/* 160 */ _CFFI_OP(_CFFI_OP_NOOP, 159),
/* 161 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 162 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(int)
/* 163 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 164 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 165 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(int, int)
/* 166 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 167 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 168 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 169 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(int, int, char const *, unsigned char, TCOD_renderer_t)
/* 170 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 171 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 172 */ _CFFI_OP(_CFFI_OP_NOOP, 64),
/* 173 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 4),
/* 174 */ _CFFI_OP(_CFFI_OP_NOOP, 142),
/* 175 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 176 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(int, int, int)
/* 177 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 178 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 179 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 180 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 181 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(int, int, int, int)
/* 182 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 183 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 184 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 185 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 186 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 187 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(int, int, int, void *, int, int)
/* 188 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 189 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 190 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 191 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 192 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 193 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 194 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 195 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(unsigned char)
/* 196 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 4),
/* 197 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 198 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(unsigned char, TCOD_color_t)
/* 199 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 4),
/* 200 */ _CFFI_OP(_CFFI_OP_NOOP, 138),
/* 201 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 202 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *)
/* 203 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 204 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 205 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, TCOD_alignment_t)
/* 206 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 207 */ _CFFI_OP(_CFFI_OP_NOOP, 63),
/* 208 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 209 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, TCOD_bkgnd_flag_t)
/* 210 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 211 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 212 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 213 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, TCOD_color_t)
/* 214 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 215 */ _CFFI_OP(_CFFI_OP_NOOP, 138),
/* 216 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 217 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, char const *)
/* 218 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 219 */ _CFFI_OP(_CFFI_OP_NOOP, 64),
/* 220 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 221 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int *, int *)
/* 222 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 223 */ _CFFI_OP(_CFFI_OP_NOOP, 159),
/* 224 */ _CFFI_OP(_CFFI_OP_NOOP, 159),
/* 225 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 226 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int)
/* 227 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 228 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 229 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 230 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int, int)
/* 231 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 232 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 233 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 234 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 235 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int, int, TCOD_bkgnd_flag_t, TCOD_alignment_t, char const *, ...)
/* 236 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 237 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 238 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 239 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 240 */ _CFFI_OP(_CFFI_OP_NOOP, 63),
/* 241 */ _CFFI_OP(_CFFI_OP_NOOP, 64),
/* 242 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 1),
/* 243 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int, int, TCOD_bkgnd_flag_t, TCOD_alignment_t, wchar_t const *, ...)
/* 244 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 245 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 246 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 247 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 248 */ _CFFI_OP(_CFFI_OP_NOOP, 63),
/* 249 */ _CFFI_OP(_CFFI_OP_NOOP, 74),
/* 250 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 1),
/* 251 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int, int, TCOD_color_t)
/* 252 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 253 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 254 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 255 */ _CFFI_OP(_CFFI_OP_NOOP, 138),
/* 256 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 257 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int, int, TCOD_color_t, TCOD_bkgnd_flag_t)
/* 258 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 259 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 260 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 261 */ _CFFI_OP(_CFFI_OP_NOOP, 138),
/* 262 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 263 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 264 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int, int, char const *, ...)
/* 265 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 266 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 267 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 268 */ _CFFI_OP(_CFFI_OP_NOOP, 64),
/* 269 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 1),
/* 270 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int, int, int)
/* 271 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 272 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 273 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 274 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 275 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 276 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int, int, int, TCOD_bkgnd_flag_t)
/* 277 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 278 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 279 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 280 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 281 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 282 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 283 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int, int, int, TCOD_color_t, TCOD_color_t)
/* 284 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 285 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 286 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 287 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 288 */ _CFFI_OP(_CFFI_OP_NOOP, 138),
/* 289 */ _CFFI_OP(_CFFI_OP_NOOP, 138),
/* 290 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 291 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int, int, int, int, int)
/* 292 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 293 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 294 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 295 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 296 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 297 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 298 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 299 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int, int, int, int, unsigned char, TCOD_bkgnd_flag_t)
/* 300 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 301 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 302 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 303 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 304 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 305 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 4),
/* 306 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 307 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 308 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int, int, int, int, unsigned char, TCOD_bkgnd_flag_t, char const *, ...)
/* 309 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 310 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 311 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 312 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 313 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 314 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 4),
/* 315 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 316 */ _CFFI_OP(_CFFI_OP_NOOP, 64),
/* 317 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 1),
/* 318 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int, int, int, int, void *, int, int, float, float)
/* 319 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 320 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 321 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 322 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 323 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 324 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 325 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 326 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 327 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 13),
/* 328 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 13),
/* 329 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 330 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, int, int, wchar_t const *, ...)
/* 331 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 332 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 333 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 334 */ _CFFI_OP(_CFFI_OP_NOOP, 74),
/* 335 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 1),
/* 336 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, void *)
/* 337 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 338 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 339 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 340 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, void *, float, float, TCOD_bkgnd_flag_t, float, float, float)
/* 341 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 342 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 343 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 13),
/* 344 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 13),
/* 345 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 346 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 13),
/* 347 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 13),
/* 348 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 13),
/* 349 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 350 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, void *, int, int, int, int, TCOD_bkgnd_flag_t)
/* 351 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 352 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 353 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 354 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 355 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 356 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 357 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 358 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 359 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void *, void *, int, int, int, int, int, int)
/* 360 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 361 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 362 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 363 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 364 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 365 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 366 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 367 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 368 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 369 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void(*)(void *))
/* 370 */ _CFFI_OP(_CFFI_OP_POINTER, 202), // void(*)(void *)
/* 371 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 372 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(void)
/* 373 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 374 */ _CFFI_OP(_CFFI_OP_FUNCTION, 400), // void()(wchar_t const *, int, int)
/* 375 */ _CFFI_OP(_CFFI_OP_NOOP, 74),
/* 376 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 377 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 378 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 379 */ _CFFI_OP(_CFFI_OP_ENUM, 3), // TCOD_event_t
/* 380 */ _CFFI_OP(_CFFI_OP_ENUM, 4), // TCOD_font_flags_t
/* 381 */ _CFFI_OP(_CFFI_OP_ENUM, 5), // TCOD_key_status_t
/* 382 */ _CFFI_OP(_CFFI_OP_STRUCT_UNION, 1), // TCOD_key_t
/* 383 */ _CFFI_OP(_CFFI_OP_STRUCT_UNION, 2), // TCOD_mouse_t
/* 384 */ _CFFI_OP(_CFFI_OP_POINTER, 385), // char *
/* 385 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 2), // char
/* 386 */ _CFFI_OP(_CFFI_OP_POINTER, 56), // int(*)(void *, int, int, int, int, TCOD_bkgnd_flag_t, TCOD_alignment_t, char const *, ...)
/* 387 */ _CFFI_OP(_CFFI_OP_POINTER, 66), // int(*)(void *, int, int, int, int, TCOD_bkgnd_flag_t, TCOD_alignment_t, wchar_t const *, ...)
/* 388 */ _CFFI_OP(_CFFI_OP_POINTER, 76), // int(*)(void *, int, int, int, int, char const *, ...)
/* 389 */ _CFFI_OP(_CFFI_OP_POINTER, 84), // int(*)(void *, int, int, int, int, wchar_t const *, ...)
/* 390 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 9), // long
/* 391 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 5), // short
/* 392 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 8), // unsigned int
/* 393 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 10), // unsigned long
/* 394 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 6), // unsigned short
/* 395 */ _CFFI_OP(_CFFI_OP_POINTER, 235), // void(*)(void *, int, int, TCOD_bkgnd_flag_t, TCOD_alignment_t, char const *, ...)
/* 396 */ _CFFI_OP(_CFFI_OP_POINTER, 243), // void(*)(void *, int, int, TCOD_bkgnd_flag_t, TCOD_alignment_t, wchar_t const *, ...)
/* 397 */ _CFFI_OP(_CFFI_OP_POINTER, 264), // void(*)(void *, int, int, char const *, ...)
/* 398 */ _CFFI_OP(_CFFI_OP_POINTER, 308), // void(*)(void *, int, int, int, int, unsigned char, TCOD_bkgnd_flag_t, char const *, ...)
/* 399 */ _CFFI_OP(_CFFI_OP_POINTER, 330), // void(*)(void *, int, int, wchar_t const *, ...)
/* 400 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 0), // void
/* 401 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 16), // wchar_t
};

static int _cffi_const_TCOD_LEFT(unsigned long long *o)
{
  int n = (TCOD_LEFT) <= 0;
  *o = (unsigned long long)((TCOD_LEFT) << 0);  /* check that TCOD_LEFT is an integer */
  return n;
}

static int _cffi_const_TCOD_RIGHT(unsigned long long *o)
{
  int n = (TCOD_RIGHT) <= 0;
  *o = (unsigned long long)((TCOD_RIGHT) << 0);  /* check that TCOD_RIGHT is an integer */
  return n;
}

static int _cffi_const_TCOD_CENTER(unsigned long long *o)
{
  int n = (TCOD_CENTER) <= 0;
  *o = (unsigned long long)((TCOD_CENTER) << 0);  /* check that TCOD_CENTER is an integer */
  return n;
}

static int _cffi_const_TCOD_BKGND_NONE(unsigned long long *o)
{
  int n = (TCOD_BKGND_NONE) <= 0;
  *o = (unsigned long long)((TCOD_BKGND_NONE) << 0);  /* check that TCOD_BKGND_NONE is an integer */
  return n;
}

static int _cffi_const_TCOD_BKGND_SET(unsigned long long *o)
{
  int n = (TCOD_BKGND_SET) <= 0;
  *o = (unsigned long long)((TCOD_BKGND_SET) << 0);  /* check that TCOD_BKGND_SET is an integer */
  return n;
}

static int _cffi_const_TCOD_BKGND_MULTIPLY(unsigned long long *o)
{
  int n = (TCOD_BKGND_MULTIPLY) <= 0;
  *o = (unsigned long long)((TCOD_BKGND_MULTIPLY) << 0);  /* check that TCOD_BKGND_MULTIPLY is an integer */
  return n;
}

static int _cffi_const_TCOD_BKGND_LIGHTEN(unsigned long long *o)
{
  int n = (TCOD_BKGND_LIGHTEN) <= 0;
  *o = (unsigned long long)((TCOD_BKGND_LIGHTEN) << 0);  /* check that TCOD_BKGND_LIGHTEN is an integer */
  return n;
}

static int _cffi_const_TCOD_BKGND_DARKEN(unsigned long long *o)
{
  int n = (TCOD_BKGND_DARKEN) <= 0;
  *o = (unsigned long long)((TCOD_BKGND_DARKEN) << 0);  /* check that TCOD_BKGND_DARKEN is an integer */
  return n;
}

static int _cffi_const_TCOD_BKGND_SCREEN(unsigned long long *o)
{
  int n = (TCOD_BKGND_SCREEN) <= 0;
  *o = (unsigned long long)((TCOD_BKGND_SCREEN) << 0);  /* check that TCOD_BKGND_SCREEN is an integer */
  return n;
}

static int _cffi_const_TCOD_BKGND_COLOR_DODGE(unsigned long long *o)
{
  int n = (TCOD_BKGND_COLOR_DODGE) <= 0;
  *o = (unsigned long long)((TCOD_BKGND_COLOR_DODGE) << 0);  /* check that TCOD_BKGND_COLOR_DODGE is an integer */
  return n;
}

static int _cffi_const_TCOD_BKGND_COLOR_BURN(unsigned long long *o)
{
  int n = (TCOD_BKGND_COLOR_BURN) <= 0;
  *o = (unsigned long long)((TCOD_BKGND_COLOR_BURN) << 0);  /* check that TCOD_BKGND_COLOR_BURN is an integer */
  return n;
}

static int _cffi_const_TCOD_BKGND_ADD(unsigned long long *o)
{
  int n = (TCOD_BKGND_ADD) <= 0;
  *o = (unsigned long long)((TCOD_BKGND_ADD) << 0);  /* check that TCOD_BKGND_ADD is an integer */
  return n;
}

static int _cffi_const_TCOD_BKGND_ADDA(unsigned long long *o)
{
  int n = (TCOD_BKGND_ADDA) <= 0;
  *o = (unsigned long long)((TCOD_BKGND_ADDA) << 0);  /* check that TCOD_BKGND_ADDA is an integer */
  return n;
}

static int _cffi_const_TCOD_BKGND_BURN(unsigned long long *o)
{
  int n = (TCOD_BKGND_BURN) <= 0;
  *o = (unsigned long long)((TCOD_BKGND_BURN) << 0);  /* check that TCOD_BKGND_BURN is an integer */
  return n;
}

static int _cffi_const_TCOD_BKGND_OVERLAY(unsigned long long *o)
{
  int n = (TCOD_BKGND_OVERLAY) <= 0;
  *o = (unsigned long long)((TCOD_BKGND_OVERLAY) << 0);  /* check that TCOD_BKGND_OVERLAY is an integer */
  return n;
}

static int _cffi_const_TCOD_BKGND_ALPH(unsigned long long *o)
{
  int n = (TCOD_BKGND_ALPH) <= 0;
  *o = (unsigned long long)((TCOD_BKGND_ALPH) << 0);  /* check that TCOD_BKGND_ALPH is an integer */
  return n;
}

static int _cffi_const_TCOD_BKGND_DEFAULT(unsigned long long *o)
{
  int n = (TCOD_BKGND_DEFAULT) <= 0;
  *o = (unsigned long long)((TCOD_BKGND_DEFAULT) << 0);  /* check that TCOD_BKGND_DEFAULT is an integer */
  return n;
}

static int _cffi_const_TCOD_COLCTRL_1(unsigned long long *o)
{
  int n = (TCOD_COLCTRL_1) <= 0;
  *o = (unsigned long long)((TCOD_COLCTRL_1) << 0);  /* check that TCOD_COLCTRL_1 is an integer */
  return n;
}

static int _cffi_const_TCOD_COLCTRL_2(unsigned long long *o)
{
  int n = (TCOD_COLCTRL_2) <= 0;
  *o = (unsigned long long)((TCOD_COLCTRL_2) << 0);  /* check that TCOD_COLCTRL_2 is an integer */
  return n;
}

static int _cffi_const_TCOD_COLCTRL_3(unsigned long long *o)
{
  int n = (TCOD_COLCTRL_3) <= 0;
  *o = (unsigned long long)((TCOD_COLCTRL_3) << 0);  /* check that TCOD_COLCTRL_3 is an integer */
  return n;
}

static int _cffi_const_TCOD_COLCTRL_4(unsigned long long *o)
{
  int n = (TCOD_COLCTRL_4) <= 0;
  *o = (unsigned long long)((TCOD_COLCTRL_4) << 0);  /* check that TCOD_COLCTRL_4 is an integer */
  return n;
}

static int _cffi_const_TCOD_COLCTRL_5(unsigned long long *o)
{
  int n = (TCOD_COLCTRL_5) <= 0;
  *o = (unsigned long long)((TCOD_COLCTRL_5) << 0);  /* check that TCOD_COLCTRL_5 is an integer */
  return n;
}

static int _cffi_const_TCOD_COLCTRL_NUMBER(unsigned long long *o)
{
  int n = (TCOD_COLCTRL_NUMBER) <= 0;
  *o = (unsigned long long)((TCOD_COLCTRL_NUMBER) << 0);  /* check that TCOD_COLCTRL_NUMBER is an integer */
  return n;
}

static int _cffi_const_TCOD_COLCTRL_FORE_RGB(unsigned long long *o)
{
  int n = (TCOD_COLCTRL_FORE_RGB) <= 0;
  *o = (unsigned long long)((TCOD_COLCTRL_FORE_RGB) << 0);  /* check that TCOD_COLCTRL_FORE_RGB is an integer */
  return n;
}

static int _cffi_const_TCOD_COLCTRL_BACK_RGB(unsigned long long *o)
{
  int n = (TCOD_COLCTRL_BACK_RGB) <= 0;
  *o = (unsigned long long)((TCOD_COLCTRL_BACK_RGB) << 0);  /* check that TCOD_COLCTRL_BACK_RGB is an integer */
  return n;
}

static int _cffi_const_TCOD_COLCTRL_STOP(unsigned long long *o)
{
  int n = (TCOD_COLCTRL_STOP) <= 0;
  *o = (unsigned long long)((TCOD_COLCTRL_STOP) << 0);  /* check that TCOD_COLCTRL_STOP is an integer */
  return n;
}

_CFFI_UNUSED_FN
static void _cffi_checkfld_typedef_TCOD_color_t(TCOD_color_t *p)
{
  /* only to generate compile-time warnings or errors */
  (void)p;
  (void)((p->r) << 1);  /* check that 'TCOD_color_t.r' is an integer */
  (void)((p->g) << 1);  /* check that 'TCOD_color_t.g' is an integer */
  (void)((p->b) << 1);  /* check that 'TCOD_color_t.b' is an integer */
}
struct _cffi_align_typedef_TCOD_color_t { char x; TCOD_color_t y; };

static int _cffi_const_TCOD_EVENT_KEY_PRESS(unsigned long long *o)
{
  int n = (TCOD_EVENT_KEY_PRESS) <= 0;
  *o = (unsigned long long)((TCOD_EVENT_KEY_PRESS) << 0);  /* check that TCOD_EVENT_KEY_PRESS is an integer */
  return n;
}

static int _cffi_const_TCOD_EVENT_KEY_RELEASE(unsigned long long *o)
{
  int n = (TCOD_EVENT_KEY_RELEASE) <= 0;
  *o = (unsigned long long)((TCOD_EVENT_KEY_RELEASE) << 0);  /* check that TCOD_EVENT_KEY_RELEASE is an integer */
  return n;
}

static int _cffi_const_TCOD_EVENT_KEY(unsigned long long *o)
{
  int n = (TCOD_EVENT_KEY) <= 0;
  *o = (unsigned long long)((TCOD_EVENT_KEY) << 0);  /* check that TCOD_EVENT_KEY is an integer */
  return n;
}

static int _cffi_const_TCOD_EVENT_MOUSE_MOVE(unsigned long long *o)
{
  int n = (TCOD_EVENT_MOUSE_MOVE) <= 0;
  *o = (unsigned long long)((TCOD_EVENT_MOUSE_MOVE) << 0);  /* check that TCOD_EVENT_MOUSE_MOVE is an integer */
  return n;
}

static int _cffi_const_TCOD_EVENT_MOUSE_PRESS(unsigned long long *o)
{
  int n = (TCOD_EVENT_MOUSE_PRESS) <= 0;
  *o = (unsigned long long)((TCOD_EVENT_MOUSE_PRESS) << 0);  /* check that TCOD_EVENT_MOUSE_PRESS is an integer */
  return n;
}

static int _cffi_const_TCOD_EVENT_MOUSE_RELEASE(unsigned long long *o)
{
  int n = (TCOD_EVENT_MOUSE_RELEASE) <= 0;
  *o = (unsigned long long)((TCOD_EVENT_MOUSE_RELEASE) << 0);  /* check that TCOD_EVENT_MOUSE_RELEASE is an integer */
  return n;
}

static int _cffi_const_TCOD_EVENT_MOUSE(unsigned long long *o)
{
  int n = (TCOD_EVENT_MOUSE) <= 0;
  *o = (unsigned long long)((TCOD_EVENT_MOUSE) << 0);  /* check that TCOD_EVENT_MOUSE is an integer */
  return n;
}

static int _cffi_const_TCOD_EVENT_ANY(unsigned long long *o)
{
  int n = (TCOD_EVENT_ANY) <= 0;
  *o = (unsigned long long)((TCOD_EVENT_ANY) << 0);  /* check that TCOD_EVENT_ANY is an integer */
  return n;
}

static int _cffi_const_TCOD_FONT_LAYOUT_ASCII_INCOL(unsigned long long *o)
{
  int n = (TCOD_FONT_LAYOUT_ASCII_INCOL) <= 0;
  *o = (unsigned long long)((TCOD_FONT_LAYOUT_ASCII_INCOL) << 0);  /* check that TCOD_FONT_LAYOUT_ASCII_INCOL is an integer */
  return n;
}

static int _cffi_const_TCOD_FONT_LAYOUT_ASCII_INROW(unsigned long long *o)
{
  int n = (TCOD_FONT_LAYOUT_ASCII_INROW) <= 0;
  *o = (unsigned long long)((TCOD_FONT_LAYOUT_ASCII_INROW) << 0);  /* check that TCOD_FONT_LAYOUT_ASCII_INROW is an integer */
  return n;
}

static int _cffi_const_TCOD_FONT_TYPE_GREYSCALE(unsigned long long *o)
{
  int n = (TCOD_FONT_TYPE_GREYSCALE) <= 0;
  *o = (unsigned long long)((TCOD_FONT_TYPE_GREYSCALE) << 0);  /* check that TCOD_FONT_TYPE_GREYSCALE is an integer */
  return n;
}

static int _cffi_const_TCOD_FONT_TYPE_GRAYSCALE(unsigned long long *o)
{
  int n = (TCOD_FONT_TYPE_GRAYSCALE) <= 0;
  *o = (unsigned long long)((TCOD_FONT_TYPE_GRAYSCALE) << 0);  /* check that TCOD_FONT_TYPE_GRAYSCALE is an integer */
  return n;
}

static int _cffi_const_TCOD_FONT_LAYOUT_TCOD(unsigned long long *o)
{
  int n = (TCOD_FONT_LAYOUT_TCOD) <= 0;
  *o = (unsigned long long)((TCOD_FONT_LAYOUT_TCOD) << 0);  /* check that TCOD_FONT_LAYOUT_TCOD is an integer */
  return n;
}

static int _cffi_const_TCOD_KEY_PRESSED(unsigned long long *o)
{
  int n = (TCOD_KEY_PRESSED) <= 0;
  *o = (unsigned long long)((TCOD_KEY_PRESSED) << 0);  /* check that TCOD_KEY_PRESSED is an integer */
  return n;
}

static int _cffi_const_TCOD_KEY_RELEASED(unsigned long long *o)
{
  int n = (TCOD_KEY_RELEASED) <= 0;
  *o = (unsigned long long)((TCOD_KEY_RELEASED) << 0);  /* check that TCOD_KEY_RELEASED is an integer */
  return n;
}

_CFFI_UNUSED_FN
static void _cffi_checkfld_typedef_TCOD_key_t(TCOD_key_t *p)
{
  /* only to generate compile-time warnings or errors */
  (void)p;
  { TCOD_keycode_t *tmp = &p->vk; (void)tmp; }
  { char *tmp = &p->c; (void)tmp; }
  (void)((p->pressed) << 1);  /* check that 'TCOD_key_t.pressed' is an integer */
  (void)((p->lalt) << 1);  /* check that 'TCOD_key_t.lalt' is an integer */
  (void)((p->lctrl) << 1);  /* check that 'TCOD_key_t.lctrl' is an integer */
  (void)((p->ralt) << 1);  /* check that 'TCOD_key_t.ralt' is an integer */
  (void)((p->rctrl) << 1);  /* check that 'TCOD_key_t.rctrl' is an integer */
  (void)((p->shift) << 1);  /* check that 'TCOD_key_t.shift' is an integer */
}
struct _cffi_align_typedef_TCOD_key_t { char x; TCOD_key_t y; };

static int _cffi_const_TCODK_NONE(unsigned long long *o)
{
  int n = (TCODK_NONE) <= 0;
  *o = (unsigned long long)((TCODK_NONE) << 0);  /* check that TCODK_NONE is an integer */
  return n;
}

static int _cffi_const_TCODK_ESCAPE(unsigned long long *o)
{
  int n = (TCODK_ESCAPE) <= 0;
  *o = (unsigned long long)((TCODK_ESCAPE) << 0);  /* check that TCODK_ESCAPE is an integer */
  return n;
}

static int _cffi_const_TCODK_BACKSPACE(unsigned long long *o)
{
  int n = (TCODK_BACKSPACE) <= 0;
  *o = (unsigned long long)((TCODK_BACKSPACE) << 0);  /* check that TCODK_BACKSPACE is an integer */
  return n;
}

static int _cffi_const_TCODK_TAB(unsigned long long *o)
{
  int n = (TCODK_TAB) <= 0;
  *o = (unsigned long long)((TCODK_TAB) << 0);  /* check that TCODK_TAB is an integer */
  return n;
}

static int _cffi_const_TCODK_ENTER(unsigned long long *o)
{
  int n = (TCODK_ENTER) <= 0;
  *o = (unsigned long long)((TCODK_ENTER) << 0);  /* check that TCODK_ENTER is an integer */
  return n;
}

static int _cffi_const_TCODK_SHIFT(unsigned long long *o)
{
  int n = (TCODK_SHIFT) <= 0;
  *o = (unsigned long long)((TCODK_SHIFT) << 0);  /* check that TCODK_SHIFT is an integer */
  return n;
}

static int _cffi_const_TCODK_CONTROL(unsigned long long *o)
{
  int n = (TCODK_CONTROL) <= 0;
  *o = (unsigned long long)((TCODK_CONTROL) << 0);  /* check that TCODK_CONTROL is an integer */
  return n;
}

static int _cffi_const_TCODK_ALT(unsigned long long *o)
{
  int n = (TCODK_ALT) <= 0;
  *o = (unsigned long long)((TCODK_ALT) << 0);  /* check that TCODK_ALT is an integer */
  return n;
}

static int _cffi_const_TCODK_PAUSE(unsigned long long *o)
{
  int n = (TCODK_PAUSE) <= 0;
  *o = (unsigned long long)((TCODK_PAUSE) << 0);  /* check that TCODK_PAUSE is an integer */
  return n;
}

static int _cffi_const_TCODK_CAPSLOCK(unsigned long long *o)
{
  int n = (TCODK_CAPSLOCK) <= 0;
  *o = (unsigned long long)((TCODK_CAPSLOCK) << 0);  /* check that TCODK_CAPSLOCK is an integer */
  return n;
}

static int _cffi_const_TCODK_PAGEUP(unsigned long long *o)
{
  int n = (TCODK_PAGEUP) <= 0;
  *o = (unsigned long long)((TCODK_PAGEUP) << 0);  /* check that TCODK_PAGEUP is an integer */
  return n;
}

static int _cffi_const_TCODK_PAGEDOWN(unsigned long long *o)
{
  int n = (TCODK_PAGEDOWN) <= 0;
  *o = (unsigned long long)((TCODK_PAGEDOWN) << 0);  /* check that TCODK_PAGEDOWN is an integer */
  return n;
}

static int _cffi_const_TCODK_END(unsigned long long *o)
{
  int n = (TCODK_END) <= 0;
  *o = (unsigned long long)((TCODK_END) << 0);  /* check that TCODK_END is an integer */
  return n;
}

static int _cffi_const_TCODK_HOME(unsigned long long *o)
{
  int n = (TCODK_HOME) <= 0;
  *o = (unsigned long long)((TCODK_HOME) << 0);  /* check that TCODK_HOME is an integer */
  return n;
}

static int _cffi_const_TCODK_UP(unsigned long long *o)
{
  int n = (TCODK_UP) <= 0;
  *o = (unsigned long long)((TCODK_UP) << 0);  /* check that TCODK_UP is an integer */
  return n;
}

static int _cffi_const_TCODK_LEFT(unsigned long long *o)
{
  int n = (TCODK_LEFT) <= 0;
  *o = (unsigned long long)((TCODK_LEFT) << 0);  /* check that TCODK_LEFT is an integer */
  return n;
}

static int _cffi_const_TCODK_RIGHT(unsigned long long *o)
{
  int n = (TCODK_RIGHT) <= 0;
  *o = (unsigned long long)((TCODK_RIGHT) << 0);  /* check that TCODK_RIGHT is an integer */
  return n;
}

static int _cffi_const_TCODK_DOWN(unsigned long long *o)
{
  int n = (TCODK_DOWN) <= 0;
  *o = (unsigned long long)((TCODK_DOWN) << 0);  /* check that TCODK_DOWN is an integer */
  return n;
}

static int _cffi_const_TCODK_PRINTSCREEN(unsigned long long *o)
{
  int n = (TCODK_PRINTSCREEN) <= 0;
  *o = (unsigned long long)((TCODK_PRINTSCREEN) << 0);  /* check that TCODK_PRINTSCREEN is an integer */
  return n;
}

static int _cffi_const_TCODK_INSERT(unsigned long long *o)
{
  int n = (TCODK_INSERT) <= 0;
  *o = (unsigned long long)((TCODK_INSERT) << 0);  /* check that TCODK_INSERT is an integer */
  return n;
}

static int _cffi_const_TCODK_DELETE(unsigned long long *o)
{
  int n = (TCODK_DELETE) <= 0;
  *o = (unsigned long long)((TCODK_DELETE) << 0);  /* check that TCODK_DELETE is an integer */
  return n;
}

static int _cffi_const_TCODK_LWIN(unsigned long long *o)
{
  int n = (TCODK_LWIN) <= 0;
  *o = (unsigned long long)((TCODK_LWIN) << 0);  /* check that TCODK_LWIN is an integer */
  return n;
}

static int _cffi_const_TCODK_RWIN(unsigned long long *o)
{
  int n = (TCODK_RWIN) <= 0;
  *o = (unsigned long long)((TCODK_RWIN) << 0);  /* check that TCODK_RWIN is an integer */
  return n;
}

static int _cffi_const_TCODK_APPS(unsigned long long *o)
{
  int n = (TCODK_APPS) <= 0;
  *o = (unsigned long long)((TCODK_APPS) << 0);  /* check that TCODK_APPS is an integer */
  return n;
}

static int _cffi_const_TCODK_0(unsigned long long *o)
{
  int n = (TCODK_0) <= 0;
  *o = (unsigned long long)((TCODK_0) << 0);  /* check that TCODK_0 is an integer */
  return n;
}

static int _cffi_const_TCODK_1(unsigned long long *o)
{
  int n = (TCODK_1) <= 0;
  *o = (unsigned long long)((TCODK_1) << 0);  /* check that TCODK_1 is an integer */
  return n;
}

static int _cffi_const_TCODK_2(unsigned long long *o)
{
  int n = (TCODK_2) <= 0;
  *o = (unsigned long long)((TCODK_2) << 0);  /* check that TCODK_2 is an integer */
  return n;
}

static int _cffi_const_TCODK_3(unsigned long long *o)
{
  int n = (TCODK_3) <= 0;
  *o = (unsigned long long)((TCODK_3) << 0);  /* check that TCODK_3 is an integer */
  return n;
}

static int _cffi_const_TCODK_4(unsigned long long *o)
{
  int n = (TCODK_4) <= 0;
  *o = (unsigned long long)((TCODK_4) << 0);  /* check that TCODK_4 is an integer */
  return n;
}

static int _cffi_const_TCODK_5(unsigned long long *o)
{
  int n = (TCODK_5) <= 0;
  *o = (unsigned long long)((TCODK_5) << 0);  /* check that TCODK_5 is an integer */
  return n;
}

static int _cffi_const_TCODK_6(unsigned long long *o)
{
  int n = (TCODK_6) <= 0;
  *o = (unsigned long long)((TCODK_6) << 0);  /* check that TCODK_6 is an integer */
  return n;
}

static int _cffi_const_TCODK_7(unsigned long long *o)
{
  int n = (TCODK_7) <= 0;
  *o = (unsigned long long)((TCODK_7) << 0);  /* check that TCODK_7 is an integer */
  return n;
}

static int _cffi_const_TCODK_8(unsigned long long *o)
{
  int n = (TCODK_8) <= 0;
  *o = (unsigned long long)((TCODK_8) << 0);  /* check that TCODK_8 is an integer */
  return n;
}

static int _cffi_const_TCODK_9(unsigned long long *o)
{
  int n = (TCODK_9) <= 0;
  *o = (unsigned long long)((TCODK_9) << 0);  /* check that TCODK_9 is an integer */
  return n;
}

static int _cffi_const_TCODK_KP0(unsigned long long *o)
{
  int n = (TCODK_KP0) <= 0;
  *o = (unsigned long long)((TCODK_KP0) << 0);  /* check that TCODK_KP0 is an integer */
  return n;
}

static int _cffi_const_TCODK_KP1(unsigned long long *o)
{
  int n = (TCODK_KP1) <= 0;
  *o = (unsigned long long)((TCODK_KP1) << 0);  /* check that TCODK_KP1 is an integer */
  return n;
}

static int _cffi_const_TCODK_KP2(unsigned long long *o)
{
  int n = (TCODK_KP2) <= 0;
  *o = (unsigned long long)((TCODK_KP2) << 0);  /* check that TCODK_KP2 is an integer */
  return n;
}

static int _cffi_const_TCODK_KP3(unsigned long long *o)
{
  int n = (TCODK_KP3) <= 0;
  *o = (unsigned long long)((TCODK_KP3) << 0);  /* check that TCODK_KP3 is an integer */
  return n;
}

static int _cffi_const_TCODK_KP4(unsigned long long *o)
{
  int n = (TCODK_KP4) <= 0;
  *o = (unsigned long long)((TCODK_KP4) << 0);  /* check that TCODK_KP4 is an integer */
  return n;
}

static int _cffi_const_TCODK_KP5(unsigned long long *o)
{
  int n = (TCODK_KP5) <= 0;
  *o = (unsigned long long)((TCODK_KP5) << 0);  /* check that TCODK_KP5 is an integer */
  return n;
}

static int _cffi_const_TCODK_KP6(unsigned long long *o)
{
  int n = (TCODK_KP6) <= 0;
  *o = (unsigned long long)((TCODK_KP6) << 0);  /* check that TCODK_KP6 is an integer */
  return n;
}

static int _cffi_const_TCODK_KP7(unsigned long long *o)
{
  int n = (TCODK_KP7) <= 0;
  *o = (unsigned long long)((TCODK_KP7) << 0);  /* check that TCODK_KP7 is an integer */
  return n;
}

static int _cffi_const_TCODK_KP8(unsigned long long *o)
{
  int n = (TCODK_KP8) <= 0;
  *o = (unsigned long long)((TCODK_KP8) << 0);  /* check that TCODK_KP8 is an integer */
  return n;
}

static int _cffi_const_TCODK_KP9(unsigned long long *o)
{
  int n = (TCODK_KP9) <= 0;
  *o = (unsigned long long)((TCODK_KP9) << 0);  /* check that TCODK_KP9 is an integer */
  return n;
}

static int _cffi_const_TCODK_KPADD(unsigned long long *o)
{
  int n = (TCODK_KPADD) <= 0;
  *o = (unsigned long long)((TCODK_KPADD) << 0);  /* check that TCODK_KPADD is an integer */
  return n;
}

static int _cffi_const_TCODK_KPSUB(unsigned long long *o)
{
  int n = (TCODK_KPSUB) <= 0;
  *o = (unsigned long long)((TCODK_KPSUB) << 0);  /* check that TCODK_KPSUB is an integer */
  return n;
}

static int _cffi_const_TCODK_KPDIV(unsigned long long *o)
{
  int n = (TCODK_KPDIV) <= 0;
  *o = (unsigned long long)((TCODK_KPDIV) << 0);  /* check that TCODK_KPDIV is an integer */
  return n;
}

static int _cffi_const_TCODK_KPMUL(unsigned long long *o)
{
  int n = (TCODK_KPMUL) <= 0;
  *o = (unsigned long long)((TCODK_KPMUL) << 0);  /* check that TCODK_KPMUL is an integer */
  return n;
}

static int _cffi_const_TCODK_KPDEC(unsigned long long *o)
{
  int n = (TCODK_KPDEC) <= 0;
  *o = (unsigned long long)((TCODK_KPDEC) << 0);  /* check that TCODK_KPDEC is an integer */
  return n;
}

static int _cffi_const_TCODK_KPENTER(unsigned long long *o)
{
  int n = (TCODK_KPENTER) <= 0;
  *o = (unsigned long long)((TCODK_KPENTER) << 0);  /* check that TCODK_KPENTER is an integer */
  return n;
}

static int _cffi_const_TCODK_F1(unsigned long long *o)
{
  int n = (TCODK_F1) <= 0;
  *o = (unsigned long long)((TCODK_F1) << 0);  /* check that TCODK_F1 is an integer */
  return n;
}

static int _cffi_const_TCODK_F2(unsigned long long *o)
{
  int n = (TCODK_F2) <= 0;
  *o = (unsigned long long)((TCODK_F2) << 0);  /* check that TCODK_F2 is an integer */
  return n;
}

static int _cffi_const_TCODK_F3(unsigned long long *o)
{
  int n = (TCODK_F3) <= 0;
  *o = (unsigned long long)((TCODK_F3) << 0);  /* check that TCODK_F3 is an integer */
  return n;
}

static int _cffi_const_TCODK_F4(unsigned long long *o)
{
  int n = (TCODK_F4) <= 0;
  *o = (unsigned long long)((TCODK_F4) << 0);  /* check that TCODK_F4 is an integer */
  return n;
}

static int _cffi_const_TCODK_F5(unsigned long long *o)
{
  int n = (TCODK_F5) <= 0;
  *o = (unsigned long long)((TCODK_F5) << 0);  /* check that TCODK_F5 is an integer */
  return n;
}

static int _cffi_const_TCODK_F6(unsigned long long *o)
{
  int n = (TCODK_F6) <= 0;
  *o = (unsigned long long)((TCODK_F6) << 0);  /* check that TCODK_F6 is an integer */
  return n;
}

static int _cffi_const_TCODK_F7(unsigned long long *o)
{
  int n = (TCODK_F7) <= 0;
  *o = (unsigned long long)((TCODK_F7) << 0);  /* check that TCODK_F7 is an integer */
  return n;
}

static int _cffi_const_TCODK_F8(unsigned long long *o)
{
  int n = (TCODK_F8) <= 0;
  *o = (unsigned long long)((TCODK_F8) << 0);  /* check that TCODK_F8 is an integer */
  return n;
}

static int _cffi_const_TCODK_F9(unsigned long long *o)
{
  int n = (TCODK_F9) <= 0;
  *o = (unsigned long long)((TCODK_F9) << 0);  /* check that TCODK_F9 is an integer */
  return n;
}

static int _cffi_const_TCODK_F10(unsigned long long *o)
{
  int n = (TCODK_F10) <= 0;
  *o = (unsigned long long)((TCODK_F10) << 0);  /* check that TCODK_F10 is an integer */
  return n;
}

static int _cffi_const_TCODK_F11(unsigned long long *o)
{
  int n = (TCODK_F11) <= 0;
  *o = (unsigned long long)((TCODK_F11) << 0);  /* check that TCODK_F11 is an integer */
  return n;
}

static int _cffi_const_TCODK_F12(unsigned long long *o)
{
  int n = (TCODK_F12) <= 0;
  *o = (unsigned long long)((TCODK_F12) << 0);  /* check that TCODK_F12 is an integer */
  return n;
}

static int _cffi_const_TCODK_NUMLOCK(unsigned long long *o)
{
  int n = (TCODK_NUMLOCK) <= 0;
  *o = (unsigned long long)((TCODK_NUMLOCK) << 0);  /* check that TCODK_NUMLOCK is an integer */
  return n;
}

static int _cffi_const_TCODK_SCROLLLOCK(unsigned long long *o)
{
  int n = (TCODK_SCROLLLOCK) <= 0;
  *o = (unsigned long long)((TCODK_SCROLLLOCK) << 0);  /* check that TCODK_SCROLLLOCK is an integer */
  return n;
}

static int _cffi_const_TCODK_SPACE(unsigned long long *o)
{
  int n = (TCODK_SPACE) <= 0;
  *o = (unsigned long long)((TCODK_SPACE) << 0);  /* check that TCODK_SPACE is an integer */
  return n;
}

static int _cffi_const_TCODK_CHAR(unsigned long long *o)
{
  int n = (TCODK_CHAR) <= 0;
  *o = (unsigned long long)((TCODK_CHAR) << 0);  /* check that TCODK_CHAR is an integer */
  return n;
}

_CFFI_UNUSED_FN
static void _cffi_checkfld_typedef_TCOD_mouse_t(TCOD_mouse_t *p)
{
  /* only to generate compile-time warnings or errors */
  (void)p;
  (void)((p->x) << 1);  /* check that 'TCOD_mouse_t.x' is an integer */
  (void)((p->y) << 1);  /* check that 'TCOD_mouse_t.y' is an integer */
  (void)((p->dx) << 1);  /* check that 'TCOD_mouse_t.dx' is an integer */
  (void)((p->dy) << 1);  /* check that 'TCOD_mouse_t.dy' is an integer */
  (void)((p->cx) << 1);  /* check that 'TCOD_mouse_t.cx' is an integer */
  (void)((p->cy) << 1);  /* check that 'TCOD_mouse_t.cy' is an integer */
  (void)((p->dcx) << 1);  /* check that 'TCOD_mouse_t.dcx' is an integer */
  (void)((p->dcy) << 1);  /* check that 'TCOD_mouse_t.dcy' is an integer */
  (void)((p->lbutton) << 1);  /* check that 'TCOD_mouse_t.lbutton' is an integer */
  (void)((p->rbutton) << 1);  /* check that 'TCOD_mouse_t.rbutton' is an integer */
  (void)((p->mbutton) << 1);  /* check that 'TCOD_mouse_t.mbutton' is an integer */
  (void)((p->lbutton_pressed) << 1);  /* check that 'TCOD_mouse_t.lbutton_pressed' is an integer */
  (void)((p->rbutton_pressed) << 1);  /* check that 'TCOD_mouse_t.rbutton_pressed' is an integer */
  (void)((p->mbutton_pressed) << 1);  /* check that 'TCOD_mouse_t.mbutton_pressed' is an integer */
  (void)((p->wheel_up) << 1);  /* check that 'TCOD_mouse_t.wheel_up' is an integer */
  (void)((p->wheel_down) << 1);  /* check that 'TCOD_mouse_t.wheel_down' is an integer */
}
struct _cffi_align_typedef_TCOD_mouse_t { char x; TCOD_mouse_t y; };

static int _cffi_const_TCOD_RENDERER_GLSL(unsigned long long *o)
{
  int n = (TCOD_RENDERER_GLSL) <= 0;
  *o = (unsigned long long)((TCOD_RENDERER_GLSL) << 0);  /* check that TCOD_RENDERER_GLSL is an integer */
  return n;
}

static int _cffi_const_TCOD_RENDERER_OPENGL(unsigned long long *o)
{
  int n = (TCOD_RENDERER_OPENGL) <= 0;
  *o = (unsigned long long)((TCOD_RENDERER_OPENGL) << 0);  /* check that TCOD_RENDERER_OPENGL is an integer */
  return n;
}

static int _cffi_const_TCOD_RENDERER_SDL(unsigned long long *o)
{
  int n = (TCOD_RENDERER_SDL) <= 0;
  *o = (unsigned long long)((TCOD_RENDERER_SDL) << 0);  /* check that TCOD_RENDERER_SDL is an integer */
  return n;
}

static int _cffi_const_TCOD_NB_RENDERERS(unsigned long long *o)
{
  int n = (TCOD_NB_RENDERERS) <= 0;
  *o = (unsigned long long)((TCOD_NB_RENDERERS) << 0);  /* check that TCOD_NB_RENDERERS is an integer */
  return n;
}

static void _cffi_d_TCOD_close_library(void * x0)
{
  TCOD_close_library(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_close_library(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_close_library(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_close_library _cffi_d_TCOD_close_library
#endif

static void _cffi_d_TCOD_condition_broadcast(void * x0)
{
  TCOD_condition_broadcast(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_condition_broadcast(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_condition_broadcast(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_condition_broadcast _cffi_d_TCOD_condition_broadcast
#endif

static void _cffi_d_TCOD_condition_delete(void * x0)
{
  TCOD_condition_delete(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_condition_delete(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_condition_delete(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_condition_delete _cffi_d_TCOD_condition_delete
#endif

static void * _cffi_d_TCOD_condition_new(void)
{
  return TCOD_condition_new();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_condition_new(PyObject *self, PyObject *noarg)
{
  void * result;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_condition_new(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  return _cffi_from_c_pointer((char *)result, _cffi_type(1));
}
#else
#  define _cffi_f_TCOD_condition_new _cffi_d_TCOD_condition_new
#endif

static void _cffi_d_TCOD_condition_signal(void * x0)
{
  TCOD_condition_signal(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_condition_signal(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_condition_signal(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_condition_signal _cffi_d_TCOD_condition_signal
#endif

static void _cffi_d_TCOD_condition_wait(void * x0, void * x1)
{
  TCOD_condition_wait(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_condition_wait(PyObject *self, PyObject *args)
{
  void * x0;
  void * x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_condition_wait");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (void *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_condition_wait(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_condition_wait _cffi_d_TCOD_condition_wait
#endif

static void _cffi_d_TCOD_console_blit(void * x0, int x1, int x2, int x3, int x4, void * x5, int x6, int x7, float x8, float x9)
{
  TCOD_console_blit(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_blit(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  int x3;
  int x4;
  void * x5;
  int x6;
  int x7;
  float x8;
  float x9;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 10, "TCOD_console_blit");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, int);
  if (x3 == (int)-1 && PyErr_Occurred())
    return NULL;

  x4 = _cffi_to_c_int(arg4, int);
  if (x4 == (int)-1 && PyErr_Occurred())
    return NULL;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (void *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(1), arg5) < 0)
      return NULL;
  }

  x6 = _cffi_to_c_int(arg6, int);
  if (x6 == (int)-1 && PyErr_Occurred())
    return NULL;

  x7 = _cffi_to_c_int(arg7, int);
  if (x7 == (int)-1 && PyErr_Occurred())
    return NULL;

  x8 = (float)_cffi_to_c_float(arg8);
  if (x8 == (float)-1 && PyErr_Occurred())
    return NULL;

  x9 = (float)_cffi_to_c_float(arg9);
  if (x9 == (float)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_blit(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_blit _cffi_d_TCOD_console_blit
#endif

static TCOD_key_t _cffi_d_TCOD_console_check_for_keypress(int x0)
{
  return TCOD_console_check_for_keypress(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_check_for_keypress(PyObject *self, PyObject *arg0)
{
  int x0;
  TCOD_key_t result;

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_check_for_keypress(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_struct((char *)&result, _cffi_type(382));
}
#else
static void _cffi_f_TCOD_console_check_for_keypress(TCOD_key_t *result, int x0)
{
  { *result = TCOD_console_check_for_keypress(x0); }
}
#endif

static void _cffi_d_TCOD_console_clear(void * x0)
{
  TCOD_console_clear(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_clear(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_clear(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_clear _cffi_d_TCOD_console_clear
#endif

static void _cffi_d_TCOD_console_credits(void)
{
  TCOD_console_credits();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_credits(PyObject *self, PyObject *noarg)
{

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_credits(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_credits _cffi_d_TCOD_console_credits
#endif

static unsigned char _cffi_d_TCOD_console_credits_render(int x0, int x1, unsigned char x2)
{
  return TCOD_console_credits_render(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_credits_render(PyObject *self, PyObject *args)
{
  int x0;
  int x1;
  unsigned char x2;
  unsigned char result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "TCOD_console_credits_render");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, unsigned char);
  if (x2 == (unsigned char)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_credits_render(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, unsigned char);
}
#else
#  define _cffi_f_TCOD_console_credits_render _cffi_d_TCOD_console_credits_render
#endif

static void _cffi_d_TCOD_console_credits_reset(void)
{
  TCOD_console_credits_reset();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_credits_reset(PyObject *self, PyObject *noarg)
{

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_credits_reset(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_credits_reset _cffi_d_TCOD_console_credits_reset
#endif

static void _cffi_d_TCOD_console_delete(void * x0)
{
  TCOD_console_delete(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_delete(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_delete(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_delete _cffi_d_TCOD_console_delete
#endif

static void _cffi_d_TCOD_console_disable_keyboard_repeat(void)
{
  TCOD_console_disable_keyboard_repeat();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_disable_keyboard_repeat(PyObject *self, PyObject *noarg)
{

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_disable_keyboard_repeat(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_disable_keyboard_repeat _cffi_d_TCOD_console_disable_keyboard_repeat
#endif

static void _cffi_d_TCOD_console_flush(void)
{
  TCOD_console_flush();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_flush(PyObject *self, PyObject *noarg)
{

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_flush(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_flush _cffi_d_TCOD_console_flush
#endif

static void * _cffi_d_TCOD_console_from_file(char const * x0)
{
  return TCOD_console_from_file(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_from_file(PyObject *self, PyObject *arg0)
{
  char const * x0;
  Py_ssize_t datasize;
  void * result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char const *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(64), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_from_file(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_pointer((char *)result, _cffi_type(1));
}
#else
#  define _cffi_f_TCOD_console_from_file _cffi_d_TCOD_console_from_file
#endif

static TCOD_alignment_t _cffi_d_TCOD_console_get_alignment(void * x0)
{
  return TCOD_console_get_alignment(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_get_alignment(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;
  TCOD_alignment_t result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_get_alignment(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_deref((char *)&result, _cffi_type(63));
}
#else
#  define _cffi_f_TCOD_console_get_alignment _cffi_d_TCOD_console_get_alignment
#endif

static TCOD_bkgnd_flag_t _cffi_d_TCOD_console_get_background_flag(void * x0)
{
  return TCOD_console_get_background_flag(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_get_background_flag(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;
  TCOD_bkgnd_flag_t result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_get_background_flag(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_deref((char *)&result, _cffi_type(62));
}
#else
#  define _cffi_f_TCOD_console_get_background_flag _cffi_d_TCOD_console_get_background_flag
#endif

static int _cffi_d_TCOD_console_get_char(void * x0, int x1, int x2)
{
  return TCOD_console_get_char(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_get_char(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "TCOD_console_get_char");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_get_char(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_TCOD_console_get_char _cffi_d_TCOD_console_get_char
#endif

static TCOD_color_t _cffi_d_TCOD_console_get_char_background(void * x0, int x1, int x2)
{
  return TCOD_console_get_char_background(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_get_char_background(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  Py_ssize_t datasize;
  TCOD_color_t result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "TCOD_console_get_char_background");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_get_char_background(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_struct((char *)&result, _cffi_type(138));
}
#else
static void _cffi_f_TCOD_console_get_char_background(TCOD_color_t *result, void * x0, int x1, int x2)
{
  { *result = TCOD_console_get_char_background(x0, x1, x2); }
}
#endif

static TCOD_color_t _cffi_d_TCOD_console_get_char_foreground(void * x0, int x1, int x2)
{
  return TCOD_console_get_char_foreground(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_get_char_foreground(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  Py_ssize_t datasize;
  TCOD_color_t result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "TCOD_console_get_char_foreground");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_get_char_foreground(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_struct((char *)&result, _cffi_type(138));
}
#else
static void _cffi_f_TCOD_console_get_char_foreground(TCOD_color_t *result, void * x0, int x1, int x2)
{
  { *result = TCOD_console_get_char_foreground(x0, x1, x2); }
}
#endif

static TCOD_color_t _cffi_d_TCOD_console_get_default_background(void * x0)
{
  return TCOD_console_get_default_background(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_get_default_background(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;
  TCOD_color_t result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_get_default_background(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_struct((char *)&result, _cffi_type(138));
}
#else
static void _cffi_f_TCOD_console_get_default_background(TCOD_color_t *result, void * x0)
{
  { *result = TCOD_console_get_default_background(x0); }
}
#endif

static TCOD_color_t _cffi_d_TCOD_console_get_default_foreground(void * x0)
{
  return TCOD_console_get_default_foreground(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_get_default_foreground(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;
  TCOD_color_t result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_get_default_foreground(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_struct((char *)&result, _cffi_type(138));
}
#else
static void _cffi_f_TCOD_console_get_default_foreground(TCOD_color_t *result, void * x0)
{
  { *result = TCOD_console_get_default_foreground(x0); }
}
#endif

static unsigned char _cffi_d_TCOD_console_get_fade(void)
{
  return TCOD_console_get_fade();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_get_fade(PyObject *self, PyObject *noarg)
{
  unsigned char result;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_get_fade(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  return _cffi_from_c_int(result, unsigned char);
}
#else
#  define _cffi_f_TCOD_console_get_fade _cffi_d_TCOD_console_get_fade
#endif

static TCOD_color_t _cffi_d_TCOD_console_get_fading_color(void)
{
  return TCOD_console_get_fading_color();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_get_fading_color(PyObject *self, PyObject *noarg)
{
  TCOD_color_t result;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_get_fading_color(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  return _cffi_from_c_struct((char *)&result, _cffi_type(138));
}
#else
static void _cffi_f_TCOD_console_get_fading_color(TCOD_color_t *result)
{
  { *result = TCOD_console_get_fading_color(); }
}
#endif

static int _cffi_d_TCOD_console_get_height(void * x0)
{
  return TCOD_console_get_height(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_get_height(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;
  int result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_get_height(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_TCOD_console_get_height _cffi_d_TCOD_console_get_height
#endif

static void _cffi_const_TCOD_console_get_height_rect(char *o)
{
  *(int(* *)(void *, int, int, int, int, char const *, ...))o = TCOD_console_get_height_rect;
}

static void _cffi_const_TCOD_console_get_height_rect_utf(char *o)
{
  *(int(* *)(void *, int, int, int, int, wchar_t const *, ...))o = TCOD_console_get_height_rect_utf;
}

static int _cffi_d_TCOD_console_get_width(void * x0)
{
  return TCOD_console_get_width(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_get_width(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;
  int result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_get_width(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_TCOD_console_get_width _cffi_d_TCOD_console_get_width
#endif

static void _cffi_d_TCOD_console_hline(void * x0, int x1, int x2, int x3, TCOD_bkgnd_flag_t x4)
{
  TCOD_console_hline(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_hline(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  int x3;
  TCOD_bkgnd_flag_t x4;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "TCOD_console_hline");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, int);
  if (x3 == (int)-1 && PyErr_Occurred())
    return NULL;

  if (_cffi_to_c((char *)&x4, _cffi_type(62), arg4) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_hline(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_hline _cffi_d_TCOD_console_hline
#endif

static void _cffi_d_TCOD_console_init_root(int x0, int x1, char const * x2, unsigned char x3, TCOD_renderer_t x4)
{
  TCOD_console_init_root(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_init_root(PyObject *self, PyObject *args)
{
  int x0;
  int x1;
  char const * x2;
  unsigned char x3;
  TCOD_renderer_t x4;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "TCOD_console_init_root");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (char const *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(64), arg2) < 0)
      return NULL;
  }

  x3 = _cffi_to_c_int(arg3, unsigned char);
  if (x3 == (unsigned char)-1 && PyErr_Occurred())
    return NULL;

  if (_cffi_to_c((char *)&x4, _cffi_type(142), arg4) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_init_root(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_init_root _cffi_d_TCOD_console_init_root
#endif

static unsigned char _cffi_d_TCOD_console_is_fullscreen(void)
{
  return TCOD_console_is_fullscreen();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_is_fullscreen(PyObject *self, PyObject *noarg)
{
  unsigned char result;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_is_fullscreen(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  return _cffi_from_c_int(result, unsigned char);
}
#else
#  define _cffi_f_TCOD_console_is_fullscreen _cffi_d_TCOD_console_is_fullscreen
#endif

static unsigned char _cffi_d_TCOD_console_is_key_pressed(TCOD_keycode_t x0)
{
  return TCOD_console_is_key_pressed(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_is_key_pressed(PyObject *self, PyObject *arg0)
{
  TCOD_keycode_t x0;
  unsigned char result;

  if (_cffi_to_c((char *)&x0, _cffi_type(95), arg0) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_is_key_pressed(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, unsigned char);
}
#else
#  define _cffi_f_TCOD_console_is_key_pressed _cffi_d_TCOD_console_is_key_pressed
#endif

static unsigned char _cffi_d_TCOD_console_is_window_closed(void)
{
  return TCOD_console_is_window_closed();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_is_window_closed(PyObject *self, PyObject *noarg)
{
  unsigned char result;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_is_window_closed(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  return _cffi_from_c_int(result, unsigned char);
}
#else
#  define _cffi_f_TCOD_console_is_window_closed _cffi_d_TCOD_console_is_window_closed
#endif

static unsigned char _cffi_d_TCOD_console_load_apf(void * x0, char const * x1)
{
  return TCOD_console_load_apf(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_load_apf(PyObject *self, PyObject *args)
{
  void * x0;
  char const * x1;
  Py_ssize_t datasize;
  unsigned char result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_console_load_apf");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char const *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(64), arg1) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_load_apf(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, unsigned char);
}
#else
#  define _cffi_f_TCOD_console_load_apf _cffi_d_TCOD_console_load_apf
#endif

static unsigned char _cffi_d_TCOD_console_load_asc(void * x0, char const * x1)
{
  return TCOD_console_load_asc(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_load_asc(PyObject *self, PyObject *args)
{
  void * x0;
  char const * x1;
  Py_ssize_t datasize;
  unsigned char result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_console_load_asc");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char const *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(64), arg1) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_load_asc(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, unsigned char);
}
#else
#  define _cffi_f_TCOD_console_load_asc _cffi_d_TCOD_console_load_asc
#endif

static void _cffi_d_TCOD_console_map_ascii_code_to_font(int x0, int x1, int x2)
{
  TCOD_console_map_ascii_code_to_font(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_map_ascii_code_to_font(PyObject *self, PyObject *args)
{
  int x0;
  int x1;
  int x2;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "TCOD_console_map_ascii_code_to_font");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_map_ascii_code_to_font(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_map_ascii_code_to_font _cffi_d_TCOD_console_map_ascii_code_to_font
#endif

static void _cffi_d_TCOD_console_map_ascii_codes_to_font(int x0, int x1, int x2, int x3)
{
  TCOD_console_map_ascii_codes_to_font(x0, x1, x2, x3);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_map_ascii_codes_to_font(PyObject *self, PyObject *args)
{
  int x0;
  int x1;
  int x2;
  int x3;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 4, "TCOD_console_map_ascii_codes_to_font");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, int);
  if (x3 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_map_ascii_codes_to_font(x0, x1, x2, x3); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_map_ascii_codes_to_font _cffi_d_TCOD_console_map_ascii_codes_to_font
#endif

static void _cffi_d_TCOD_console_map_string_to_font(char const * x0, int x1, int x2)
{
  TCOD_console_map_string_to_font(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_map_string_to_font(PyObject *self, PyObject *args)
{
  char const * x0;
  int x1;
  int x2;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "TCOD_console_map_string_to_font");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char const *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(64), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_map_string_to_font(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_map_string_to_font _cffi_d_TCOD_console_map_string_to_font
#endif

static void _cffi_d_TCOD_console_map_string_to_font_utf(wchar_t const * x0, int x1, int x2)
{
  TCOD_console_map_string_to_font_utf(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_map_string_to_font_utf(PyObject *self, PyObject *args)
{
  wchar_t const * x0;
  int x1;
  int x2;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "TCOD_console_map_string_to_font_utf");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(74), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (wchar_t const *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(74), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_map_string_to_font_utf(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_map_string_to_font_utf _cffi_d_TCOD_console_map_string_to_font_utf
#endif

static void * _cffi_d_TCOD_console_new(int x0, int x1)
{
  return TCOD_console_new(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_new(PyObject *self, PyObject *args)
{
  int x0;
  int x1;
  void * result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_console_new");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_new(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_pointer((char *)result, _cffi_type(1));
}
#else
#  define _cffi_f_TCOD_console_new _cffi_d_TCOD_console_new
#endif

static void _cffi_const_TCOD_console_print(char *o)
{
  *(void(* *)(void *, int, int, char const *, ...))o = TCOD_console_print;
}

static void _cffi_const_TCOD_console_print_ex(char *o)
{
  *(void(* *)(void *, int, int, TCOD_bkgnd_flag_t, TCOD_alignment_t, char const *, ...))o = TCOD_console_print_ex;
}

static void _cffi_const_TCOD_console_print_ex_utf(char *o)
{
  *(void(* *)(void *, int, int, TCOD_bkgnd_flag_t, TCOD_alignment_t, wchar_t const *, ...))o = TCOD_console_print_ex_utf;
}

static void _cffi_const_TCOD_console_print_frame(char *o)
{
  *(void(* *)(void *, int, int, int, int, unsigned char, TCOD_bkgnd_flag_t, char const *, ...))o = TCOD_console_print_frame;
}

static void _cffi_const_TCOD_console_print_rect(char *o)
{
  *(int(* *)(void *, int, int, int, int, char const *, ...))o = TCOD_console_print_rect;
}

static void _cffi_const_TCOD_console_print_rect_ex(char *o)
{
  *(int(* *)(void *, int, int, int, int, TCOD_bkgnd_flag_t, TCOD_alignment_t, char const *, ...))o = TCOD_console_print_rect_ex;
}

static void _cffi_const_TCOD_console_print_rect_ex_utf(char *o)
{
  *(int(* *)(void *, int, int, int, int, TCOD_bkgnd_flag_t, TCOD_alignment_t, wchar_t const *, ...))o = TCOD_console_print_rect_ex_utf;
}

static void _cffi_const_TCOD_console_print_rect_utf(char *o)
{
  *(int(* *)(void *, int, int, int, int, wchar_t const *, ...))o = TCOD_console_print_rect_utf;
}

static void _cffi_const_TCOD_console_print_utf(char *o)
{
  *(void(* *)(void *, int, int, wchar_t const *, ...))o = TCOD_console_print_utf;
}

static void _cffi_d_TCOD_console_put_char(void * x0, int x1, int x2, int x3, TCOD_bkgnd_flag_t x4)
{
  TCOD_console_put_char(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_put_char(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  int x3;
  TCOD_bkgnd_flag_t x4;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "TCOD_console_put_char");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, int);
  if (x3 == (int)-1 && PyErr_Occurred())
    return NULL;

  if (_cffi_to_c((char *)&x4, _cffi_type(62), arg4) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_put_char(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_put_char _cffi_d_TCOD_console_put_char
#endif

static void _cffi_d_TCOD_console_put_char_ex(void * x0, int x1, int x2, int x3, TCOD_color_t x4, TCOD_color_t x5)
{
  TCOD_console_put_char_ex(x0, x1, x2, x3, x4, x5);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_put_char_ex(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  int x3;
  TCOD_color_t x4;
  TCOD_color_t x5;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 6, "TCOD_console_put_char_ex");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, int);
  if (x3 == (int)-1 && PyErr_Occurred())
    return NULL;

  if (_cffi_to_c((char *)&x4, _cffi_type(138), arg4) < 0)
    return NULL;

  if (_cffi_to_c((char *)&x5, _cffi_type(138), arg5) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_put_char_ex(x0, x1, x2, x3, x4, x5); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
static void _cffi_f_TCOD_console_put_char_ex(void * x0, int x1, int x2, int x3, TCOD_color_t *x4, TCOD_color_t *x5)
{
  { TCOD_console_put_char_ex(x0, x1, x2, x3, *x4, *x5); }
}
#endif

static void _cffi_d_TCOD_console_rect(void * x0, int x1, int x2, int x3, int x4, unsigned char x5, TCOD_bkgnd_flag_t x6)
{
  TCOD_console_rect(x0, x1, x2, x3, x4, x5, x6);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_rect(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  int x3;
  int x4;
  unsigned char x5;
  TCOD_bkgnd_flag_t x6;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 7, "TCOD_console_rect");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, int);
  if (x3 == (int)-1 && PyErr_Occurred())
    return NULL;

  x4 = _cffi_to_c_int(arg4, int);
  if (x4 == (int)-1 && PyErr_Occurred())
    return NULL;

  x5 = _cffi_to_c_int(arg5, unsigned char);
  if (x5 == (unsigned char)-1 && PyErr_Occurred())
    return NULL;

  if (_cffi_to_c((char *)&x6, _cffi_type(62), arg6) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_rect(x0, x1, x2, x3, x4, x5, x6); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_rect _cffi_d_TCOD_console_rect
#endif

static unsigned char _cffi_d_TCOD_console_save_apf(void * x0, char const * x1)
{
  return TCOD_console_save_apf(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_save_apf(PyObject *self, PyObject *args)
{
  void * x0;
  char const * x1;
  Py_ssize_t datasize;
  unsigned char result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_console_save_apf");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char const *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(64), arg1) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_save_apf(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, unsigned char);
}
#else
#  define _cffi_f_TCOD_console_save_apf _cffi_d_TCOD_console_save_apf
#endif

static unsigned char _cffi_d_TCOD_console_save_asc(void * x0, char const * x1)
{
  return TCOD_console_save_asc(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_save_asc(PyObject *self, PyObject *args)
{
  void * x0;
  char const * x1;
  Py_ssize_t datasize;
  unsigned char result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_console_save_asc");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char const *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(64), arg1) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_save_asc(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, unsigned char);
}
#else
#  define _cffi_f_TCOD_console_save_asc _cffi_d_TCOD_console_save_asc
#endif

static void _cffi_d_TCOD_console_set_alignment(void * x0, TCOD_alignment_t x1)
{
  TCOD_console_set_alignment(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_alignment(PyObject *self, PyObject *args)
{
  void * x0;
  TCOD_alignment_t x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_console_set_alignment");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  if (_cffi_to_c((char *)&x1, _cffi_type(63), arg1) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_alignment(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_set_alignment _cffi_d_TCOD_console_set_alignment
#endif

static void _cffi_d_TCOD_console_set_background_flag(void * x0, TCOD_bkgnd_flag_t x1)
{
  TCOD_console_set_background_flag(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_background_flag(PyObject *self, PyObject *args)
{
  void * x0;
  TCOD_bkgnd_flag_t x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_console_set_background_flag");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  if (_cffi_to_c((char *)&x1, _cffi_type(62), arg1) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_background_flag(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_set_background_flag _cffi_d_TCOD_console_set_background_flag
#endif

static void _cffi_d_TCOD_console_set_char(void * x0, int x1, int x2, int x3)
{
  TCOD_console_set_char(x0, x1, x2, x3);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_char(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  int x3;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 4, "TCOD_console_set_char");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, int);
  if (x3 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_char(x0, x1, x2, x3); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_set_char _cffi_d_TCOD_console_set_char
#endif

static void _cffi_d_TCOD_console_set_char_background(void * x0, int x1, int x2, TCOD_color_t x3, TCOD_bkgnd_flag_t x4)
{
  TCOD_console_set_char_background(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_char_background(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  TCOD_color_t x3;
  TCOD_bkgnd_flag_t x4;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "TCOD_console_set_char_background");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  if (_cffi_to_c((char *)&x3, _cffi_type(138), arg3) < 0)
    return NULL;

  if (_cffi_to_c((char *)&x4, _cffi_type(62), arg4) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_char_background(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
static void _cffi_f_TCOD_console_set_char_background(void * x0, int x1, int x2, TCOD_color_t *x3, TCOD_bkgnd_flag_t x4)
{
  { TCOD_console_set_char_background(x0, x1, x2, *x3, x4); }
}
#endif

static void _cffi_d_TCOD_console_set_char_foreground(void * x0, int x1, int x2, TCOD_color_t x3)
{
  TCOD_console_set_char_foreground(x0, x1, x2, x3);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_char_foreground(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  TCOD_color_t x3;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 4, "TCOD_console_set_char_foreground");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  if (_cffi_to_c((char *)&x3, _cffi_type(138), arg3) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_char_foreground(x0, x1, x2, x3); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
static void _cffi_f_TCOD_console_set_char_foreground(void * x0, int x1, int x2, TCOD_color_t *x3)
{
  { TCOD_console_set_char_foreground(x0, x1, x2, *x3); }
}
#endif

static void _cffi_d_TCOD_console_set_color_control(TCOD_colctrl_t x0, TCOD_color_t x1, TCOD_color_t x2)
{
  TCOD_console_set_color_control(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_color_control(PyObject *self, PyObject *args)
{
  TCOD_colctrl_t x0;
  TCOD_color_t x1;
  TCOD_color_t x2;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "TCOD_console_set_color_control");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  if (_cffi_to_c((char *)&x0, _cffi_type(137), arg0) < 0)
    return NULL;

  if (_cffi_to_c((char *)&x1, _cffi_type(138), arg1) < 0)
    return NULL;

  if (_cffi_to_c((char *)&x2, _cffi_type(138), arg2) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_color_control(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
static void _cffi_f_TCOD_console_set_color_control(TCOD_colctrl_t x0, TCOD_color_t *x1, TCOD_color_t *x2)
{
  { TCOD_console_set_color_control(x0, *x1, *x2); }
}
#endif

static void _cffi_d_TCOD_console_set_custom_font(char const * x0, int x1, int x2, int x3)
{
  TCOD_console_set_custom_font(x0, x1, x2, x3);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_custom_font(PyObject *self, PyObject *args)
{
  char const * x0;
  int x1;
  int x2;
  int x3;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 4, "TCOD_console_set_custom_font");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char const *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(64), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, int);
  if (x3 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_custom_font(x0, x1, x2, x3); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_set_custom_font _cffi_d_TCOD_console_set_custom_font
#endif

static void _cffi_d_TCOD_console_set_default_background(void * x0, TCOD_color_t x1)
{
  TCOD_console_set_default_background(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_default_background(PyObject *self, PyObject *args)
{
  void * x0;
  TCOD_color_t x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_console_set_default_background");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  if (_cffi_to_c((char *)&x1, _cffi_type(138), arg1) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_default_background(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
static void _cffi_f_TCOD_console_set_default_background(void * x0, TCOD_color_t *x1)
{
  { TCOD_console_set_default_background(x0, *x1); }
}
#endif

static void _cffi_d_TCOD_console_set_default_foreground(void * x0, TCOD_color_t x1)
{
  TCOD_console_set_default_foreground(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_default_foreground(PyObject *self, PyObject *args)
{
  void * x0;
  TCOD_color_t x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_console_set_default_foreground");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  if (_cffi_to_c((char *)&x1, _cffi_type(138), arg1) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_default_foreground(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
static void _cffi_f_TCOD_console_set_default_foreground(void * x0, TCOD_color_t *x1)
{
  { TCOD_console_set_default_foreground(x0, *x1); }
}
#endif

static void _cffi_d_TCOD_console_set_dirty(int x0, int x1, int x2, int x3)
{
  TCOD_console_set_dirty(x0, x1, x2, x3);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_dirty(PyObject *self, PyObject *args)
{
  int x0;
  int x1;
  int x2;
  int x3;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 4, "TCOD_console_set_dirty");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, int);
  if (x3 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_dirty(x0, x1, x2, x3); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_set_dirty _cffi_d_TCOD_console_set_dirty
#endif

static void _cffi_d_TCOD_console_set_fade(unsigned char x0, TCOD_color_t x1)
{
  TCOD_console_set_fade(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_fade(PyObject *self, PyObject *args)
{
  unsigned char x0;
  TCOD_color_t x1;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_console_set_fade");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  x0 = _cffi_to_c_int(arg0, unsigned char);
  if (x0 == (unsigned char)-1 && PyErr_Occurred())
    return NULL;

  if (_cffi_to_c((char *)&x1, _cffi_type(138), arg1) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_fade(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
static void _cffi_f_TCOD_console_set_fade(unsigned char x0, TCOD_color_t *x1)
{
  { TCOD_console_set_fade(x0, *x1); }
}
#endif

static void _cffi_d_TCOD_console_set_fullscreen(unsigned char x0)
{
  TCOD_console_set_fullscreen(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_fullscreen(PyObject *self, PyObject *arg0)
{
  unsigned char x0;

  x0 = _cffi_to_c_int(arg0, unsigned char);
  if (x0 == (unsigned char)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_fullscreen(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_set_fullscreen _cffi_d_TCOD_console_set_fullscreen
#endif

static void _cffi_d_TCOD_console_set_key_color(void * x0, TCOD_color_t x1)
{
  TCOD_console_set_key_color(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_key_color(PyObject *self, PyObject *args)
{
  void * x0;
  TCOD_color_t x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_console_set_key_color");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  if (_cffi_to_c((char *)&x1, _cffi_type(138), arg1) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_key_color(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
static void _cffi_f_TCOD_console_set_key_color(void * x0, TCOD_color_t *x1)
{
  { TCOD_console_set_key_color(x0, *x1); }
}
#endif

static void _cffi_d_TCOD_console_set_keyboard_repeat(int x0, int x1)
{
  TCOD_console_set_keyboard_repeat(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_keyboard_repeat(PyObject *self, PyObject *args)
{
  int x0;
  int x1;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_console_set_keyboard_repeat");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_keyboard_repeat(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_set_keyboard_repeat _cffi_d_TCOD_console_set_keyboard_repeat
#endif

static void _cffi_d_TCOD_console_set_window_title(char const * x0)
{
  TCOD_console_set_window_title(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_set_window_title(PyObject *self, PyObject *arg0)
{
  char const * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char const *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(64), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_set_window_title(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_set_window_title _cffi_d_TCOD_console_set_window_title
#endif

static void _cffi_d_TCOD_console_vline(void * x0, int x1, int x2, int x3, TCOD_bkgnd_flag_t x4)
{
  TCOD_console_vline(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_vline(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  int x3;
  TCOD_bkgnd_flag_t x4;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "TCOD_console_vline");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, int);
  if (x3 == (int)-1 && PyErr_Occurred())
    return NULL;

  if (_cffi_to_c((char *)&x4, _cffi_type(62), arg4) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_console_vline(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_console_vline _cffi_d_TCOD_console_vline
#endif

static TCOD_key_t _cffi_d_TCOD_console_wait_for_keypress(unsigned char x0)
{
  return TCOD_console_wait_for_keypress(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_console_wait_for_keypress(PyObject *self, PyObject *arg0)
{
  unsigned char x0;
  TCOD_key_t result;

  x0 = _cffi_to_c_int(arg0, unsigned char);
  if (x0 == (unsigned char)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_console_wait_for_keypress(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_struct((char *)&result, _cffi_type(382));
}
#else
static void _cffi_f_TCOD_console_wait_for_keypress(TCOD_key_t *result, unsigned char x0)
{
  { *result = TCOD_console_wait_for_keypress(x0); }
}
#endif

static void * _cffi_d_TCOD_get_function_address(void * x0, char const * x1)
{
  return TCOD_get_function_address(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_get_function_address(PyObject *self, PyObject *args)
{
  void * x0;
  char const * x1;
  Py_ssize_t datasize;
  void * result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_get_function_address");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char const *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(64), arg1) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_get_function_address(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_pointer((char *)result, _cffi_type(1));
}
#else
#  define _cffi_f_TCOD_get_function_address _cffi_d_TCOD_get_function_address
#endif

static void _cffi_d_TCOD_image_blit(void * x0, void * x1, float x2, float x3, TCOD_bkgnd_flag_t x4, float x5, float x6, float x7)
{
  TCOD_image_blit(x0, x1, x2, x3, x4, x5, x6, x7);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_blit(PyObject *self, PyObject *args)
{
  void * x0;
  void * x1;
  float x2;
  float x3;
  TCOD_bkgnd_flag_t x4;
  float x5;
  float x6;
  float x7;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 8, "TCOD_image_blit");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (void *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  x2 = (float)_cffi_to_c_float(arg2);
  if (x2 == (float)-1 && PyErr_Occurred())
    return NULL;

  x3 = (float)_cffi_to_c_float(arg3);
  if (x3 == (float)-1 && PyErr_Occurred())
    return NULL;

  if (_cffi_to_c((char *)&x4, _cffi_type(62), arg4) < 0)
    return NULL;

  x5 = (float)_cffi_to_c_float(arg5);
  if (x5 == (float)-1 && PyErr_Occurred())
    return NULL;

  x6 = (float)_cffi_to_c_float(arg6);
  if (x6 == (float)-1 && PyErr_Occurred())
    return NULL;

  x7 = (float)_cffi_to_c_float(arg7);
  if (x7 == (float)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_blit(x0, x1, x2, x3, x4, x5, x6, x7); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_image_blit _cffi_d_TCOD_image_blit
#endif

static void _cffi_d_TCOD_image_blit_2x(void * x0, void * x1, int x2, int x3, int x4, int x5, int x6, int x7)
{
  TCOD_image_blit_2x(x0, x1, x2, x3, x4, x5, x6, x7);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_blit_2x(PyObject *self, PyObject *args)
{
  void * x0;
  void * x1;
  int x2;
  int x3;
  int x4;
  int x5;
  int x6;
  int x7;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 8, "TCOD_image_blit_2x");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (void *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, int);
  if (x3 == (int)-1 && PyErr_Occurred())
    return NULL;

  x4 = _cffi_to_c_int(arg4, int);
  if (x4 == (int)-1 && PyErr_Occurred())
    return NULL;

  x5 = _cffi_to_c_int(arg5, int);
  if (x5 == (int)-1 && PyErr_Occurred())
    return NULL;

  x6 = _cffi_to_c_int(arg6, int);
  if (x6 == (int)-1 && PyErr_Occurred())
    return NULL;

  x7 = _cffi_to_c_int(arg7, int);
  if (x7 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_blit_2x(x0, x1, x2, x3, x4, x5, x6, x7); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_image_blit_2x _cffi_d_TCOD_image_blit_2x
#endif

static void _cffi_d_TCOD_image_blit_rect(void * x0, void * x1, int x2, int x3, int x4, int x5, TCOD_bkgnd_flag_t x6)
{
  TCOD_image_blit_rect(x0, x1, x2, x3, x4, x5, x6);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_blit_rect(PyObject *self, PyObject *args)
{
  void * x0;
  void * x1;
  int x2;
  int x3;
  int x4;
  int x5;
  TCOD_bkgnd_flag_t x6;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 7, "TCOD_image_blit_rect");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (void *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, int);
  if (x3 == (int)-1 && PyErr_Occurred())
    return NULL;

  x4 = _cffi_to_c_int(arg4, int);
  if (x4 == (int)-1 && PyErr_Occurred())
    return NULL;

  x5 = _cffi_to_c_int(arg5, int);
  if (x5 == (int)-1 && PyErr_Occurred())
    return NULL;

  if (_cffi_to_c((char *)&x6, _cffi_type(62), arg6) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_blit_rect(x0, x1, x2, x3, x4, x5, x6); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_image_blit_rect _cffi_d_TCOD_image_blit_rect
#endif

static void _cffi_d_TCOD_image_clear(void * x0, TCOD_color_t x1)
{
  TCOD_image_clear(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_clear(PyObject *self, PyObject *args)
{
  void * x0;
  TCOD_color_t x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_image_clear");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  if (_cffi_to_c((char *)&x1, _cffi_type(138), arg1) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_clear(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
static void _cffi_f_TCOD_image_clear(void * x0, TCOD_color_t *x1)
{
  { TCOD_image_clear(x0, *x1); }
}
#endif

static void _cffi_d_TCOD_image_delete(void * x0)
{
  TCOD_image_delete(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_delete(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_delete(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_image_delete _cffi_d_TCOD_image_delete
#endif

static void * _cffi_d_TCOD_image_from_console(void * x0)
{
  return TCOD_image_from_console(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_from_console(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;
  void * result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_image_from_console(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_pointer((char *)result, _cffi_type(1));
}
#else
#  define _cffi_f_TCOD_image_from_console _cffi_d_TCOD_image_from_console
#endif

static int _cffi_d_TCOD_image_get_alpha(void * x0, int x1, int x2)
{
  return TCOD_image_get_alpha(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_get_alpha(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "TCOD_image_get_alpha");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_image_get_alpha(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_TCOD_image_get_alpha _cffi_d_TCOD_image_get_alpha
#endif

static TCOD_color_t _cffi_d_TCOD_image_get_mipmap_pixel(void * x0, float x1, float x2, float x3, float x4)
{
  return TCOD_image_get_mipmap_pixel(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_get_mipmap_pixel(PyObject *self, PyObject *args)
{
  void * x0;
  float x1;
  float x2;
  float x3;
  float x4;
  Py_ssize_t datasize;
  TCOD_color_t result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "TCOD_image_get_mipmap_pixel");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (float)_cffi_to_c_float(arg1);
  if (x1 == (float)-1 && PyErr_Occurred())
    return NULL;

  x2 = (float)_cffi_to_c_float(arg2);
  if (x2 == (float)-1 && PyErr_Occurred())
    return NULL;

  x3 = (float)_cffi_to_c_float(arg3);
  if (x3 == (float)-1 && PyErr_Occurred())
    return NULL;

  x4 = (float)_cffi_to_c_float(arg4);
  if (x4 == (float)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_image_get_mipmap_pixel(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_struct((char *)&result, _cffi_type(138));
}
#else
static void _cffi_f_TCOD_image_get_mipmap_pixel(TCOD_color_t *result, void * x0, float x1, float x2, float x3, float x4)
{
  { *result = TCOD_image_get_mipmap_pixel(x0, x1, x2, x3, x4); }
}
#endif

static TCOD_color_t _cffi_d_TCOD_image_get_pixel(void * x0, int x1, int x2)
{
  return TCOD_image_get_pixel(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_get_pixel(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  Py_ssize_t datasize;
  TCOD_color_t result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "TCOD_image_get_pixel");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_image_get_pixel(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_struct((char *)&result, _cffi_type(138));
}
#else
static void _cffi_f_TCOD_image_get_pixel(TCOD_color_t *result, void * x0, int x1, int x2)
{
  { *result = TCOD_image_get_pixel(x0, x1, x2); }
}
#endif

static void _cffi_d_TCOD_image_get_size(void * x0, int * x1, int * x2)
{
  TCOD_image_get_size(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_get_size(PyObject *self, PyObject *args)
{
  void * x0;
  int * x1;
  int * x2;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "TCOD_image_get_size");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(159), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(159), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(159), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(159), arg2) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_get_size(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_image_get_size _cffi_d_TCOD_image_get_size
#endif

static void _cffi_d_TCOD_image_hflip(void * x0)
{
  TCOD_image_hflip(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_hflip(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_hflip(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_image_hflip _cffi_d_TCOD_image_hflip
#endif

static void _cffi_d_TCOD_image_invert(void * x0)
{
  TCOD_image_invert(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_invert(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_invert(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_image_invert _cffi_d_TCOD_image_invert
#endif

static unsigned char _cffi_d_TCOD_image_is_pixel_transparent(void * x0, int x1, int x2)
{
  return TCOD_image_is_pixel_transparent(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_is_pixel_transparent(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  Py_ssize_t datasize;
  unsigned char result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "TCOD_image_is_pixel_transparent");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_image_is_pixel_transparent(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, unsigned char);
}
#else
#  define _cffi_f_TCOD_image_is_pixel_transparent _cffi_d_TCOD_image_is_pixel_transparent
#endif

static void * _cffi_d_TCOD_image_load(char const * x0)
{
  return TCOD_image_load(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_load(PyObject *self, PyObject *arg0)
{
  char const * x0;
  Py_ssize_t datasize;
  void * result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char const *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(64), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_image_load(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_pointer((char *)result, _cffi_type(1));
}
#else
#  define _cffi_f_TCOD_image_load _cffi_d_TCOD_image_load
#endif

static void * _cffi_d_TCOD_image_new(int x0, int x1)
{
  return TCOD_image_new(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_new(PyObject *self, PyObject *args)
{
  int x0;
  int x1;
  void * result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_image_new");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_image_new(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_pointer((char *)result, _cffi_type(1));
}
#else
#  define _cffi_f_TCOD_image_new _cffi_d_TCOD_image_new
#endif

static void _cffi_d_TCOD_image_put_pixel(void * x0, int x1, int x2, TCOD_color_t x3)
{
  TCOD_image_put_pixel(x0, x1, x2, x3);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_put_pixel(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  TCOD_color_t x3;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 4, "TCOD_image_put_pixel");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  if (_cffi_to_c((char *)&x3, _cffi_type(138), arg3) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_put_pixel(x0, x1, x2, x3); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
static void _cffi_f_TCOD_image_put_pixel(void * x0, int x1, int x2, TCOD_color_t *x3)
{
  { TCOD_image_put_pixel(x0, x1, x2, *x3); }
}
#endif

static void _cffi_d_TCOD_image_refresh_console(void * x0, void * x1)
{
  TCOD_image_refresh_console(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_refresh_console(PyObject *self, PyObject *args)
{
  void * x0;
  void * x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_image_refresh_console");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (void *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_refresh_console(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_image_refresh_console _cffi_d_TCOD_image_refresh_console
#endif

static void _cffi_d_TCOD_image_rotate90(void * x0, int x1)
{
  TCOD_image_rotate90(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_rotate90(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_image_rotate90");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_rotate90(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_image_rotate90 _cffi_d_TCOD_image_rotate90
#endif

static void _cffi_d_TCOD_image_save(void * x0, char const * x1)
{
  TCOD_image_save(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_save(PyObject *self, PyObject *args)
{
  void * x0;
  char const * x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_image_save");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char const *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(64), arg1) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_save(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_image_save _cffi_d_TCOD_image_save
#endif

static void _cffi_d_TCOD_image_scale(void * x0, int x1, int x2)
{
  TCOD_image_scale(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_scale(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "TCOD_image_scale");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_scale(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_image_scale _cffi_d_TCOD_image_scale
#endif

static void _cffi_d_TCOD_image_set_key_color(void * x0, TCOD_color_t x1)
{
  TCOD_image_set_key_color(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_set_key_color(PyObject *self, PyObject *args)
{
  void * x0;
  TCOD_color_t x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_image_set_key_color");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  if (_cffi_to_c((char *)&x1, _cffi_type(138), arg1) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_set_key_color(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
static void _cffi_f_TCOD_image_set_key_color(void * x0, TCOD_color_t *x1)
{
  { TCOD_image_set_key_color(x0, *x1); }
}
#endif

static void _cffi_d_TCOD_image_vflip(void * x0)
{
  TCOD_image_vflip(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_image_vflip(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_image_vflip(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_image_vflip _cffi_d_TCOD_image_vflip
#endif

static void * _cffi_d_TCOD_load_library(char const * x0)
{
  return TCOD_load_library(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_load_library(PyObject *self, PyObject *arg0)
{
  char const * x0;
  Py_ssize_t datasize;
  void * result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char const *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(64), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_load_library(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_pointer((char *)result, _cffi_type(1));
}
#else
#  define _cffi_f_TCOD_load_library _cffi_d_TCOD_load_library
#endif

static TCOD_mouse_t _cffi_d_TCOD_mouse_get_status(void)
{
  return TCOD_mouse_get_status();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_mouse_get_status(PyObject *self, PyObject *noarg)
{
  TCOD_mouse_t result;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_mouse_get_status(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  return _cffi_from_c_struct((char *)&result, _cffi_type(383));
}
#else
static void _cffi_f_TCOD_mouse_get_status(TCOD_mouse_t *result)
{
  { *result = TCOD_mouse_get_status(); }
}
#endif

static unsigned char _cffi_d_TCOD_mouse_is_cursor_visible(void)
{
  return TCOD_mouse_is_cursor_visible();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_mouse_is_cursor_visible(PyObject *self, PyObject *noarg)
{
  unsigned char result;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_mouse_is_cursor_visible(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  return _cffi_from_c_int(result, unsigned char);
}
#else
#  define _cffi_f_TCOD_mouse_is_cursor_visible _cffi_d_TCOD_mouse_is_cursor_visible
#endif

static void _cffi_d_TCOD_mouse_move(int x0, int x1)
{
  TCOD_mouse_move(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_mouse_move(PyObject *self, PyObject *args)
{
  int x0;
  int x1;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_mouse_move");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_mouse_move(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_mouse_move _cffi_d_TCOD_mouse_move
#endif

static void _cffi_d_TCOD_mouse_show_cursor(unsigned char x0)
{
  TCOD_mouse_show_cursor(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_mouse_show_cursor(PyObject *self, PyObject *arg0)
{
  unsigned char x0;

  x0 = _cffi_to_c_int(arg0, unsigned char);
  if (x0 == (unsigned char)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_mouse_show_cursor(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_mouse_show_cursor _cffi_d_TCOD_mouse_show_cursor
#endif

static void _cffi_d_TCOD_mutex_delete(void * x0)
{
  TCOD_mutex_delete(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_mutex_delete(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_mutex_delete(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_mutex_delete _cffi_d_TCOD_mutex_delete
#endif

static void _cffi_d_TCOD_mutex_in(void * x0)
{
  TCOD_mutex_in(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_mutex_in(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_mutex_in(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_mutex_in _cffi_d_TCOD_mutex_in
#endif

static void * _cffi_d_TCOD_mutex_new(void)
{
  return TCOD_mutex_new();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_mutex_new(PyObject *self, PyObject *noarg)
{
  void * result;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_mutex_new(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  return _cffi_from_c_pointer((char *)result, _cffi_type(1));
}
#else
#  define _cffi_f_TCOD_mutex_new _cffi_d_TCOD_mutex_new
#endif

static void _cffi_d_TCOD_mutex_out(void * x0)
{
  TCOD_mutex_out(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_mutex_out(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_mutex_out(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_mutex_out _cffi_d_TCOD_mutex_out
#endif

static void _cffi_d_TCOD_semaphore_delete(void * x0)
{
  TCOD_semaphore_delete(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_semaphore_delete(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_semaphore_delete(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_semaphore_delete _cffi_d_TCOD_semaphore_delete
#endif

static void _cffi_d_TCOD_semaphore_lock(void * x0)
{
  TCOD_semaphore_lock(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_semaphore_lock(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_semaphore_lock(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_semaphore_lock _cffi_d_TCOD_semaphore_lock
#endif

static void * _cffi_d_TCOD_semaphore_new(int x0)
{
  return TCOD_semaphore_new(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_semaphore_new(PyObject *self, PyObject *arg0)
{
  int x0;
  void * result;

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_semaphore_new(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_pointer((char *)result, _cffi_type(1));
}
#else
#  define _cffi_f_TCOD_semaphore_new _cffi_d_TCOD_semaphore_new
#endif

static void _cffi_d_TCOD_semaphore_unlock(void * x0)
{
  TCOD_semaphore_unlock(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_semaphore_unlock(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_semaphore_unlock(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_semaphore_unlock _cffi_d_TCOD_semaphore_unlock
#endif

static TCOD_event_t _cffi_d_TCOD_sys_check_for_event(int x0, TCOD_key_t * x1, TCOD_mouse_t * x2)
{
  return TCOD_sys_check_for_event(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_check_for_event(PyObject *self, PyObject *args)
{
  int x0;
  TCOD_key_t * x1;
  TCOD_mouse_t * x2;
  Py_ssize_t datasize;
  TCOD_event_t result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "TCOD_sys_check_for_event");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(25), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (TCOD_key_t *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(25), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(26), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (TCOD_mouse_t *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(26), arg2) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_sys_check_for_event(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_deref((char *)&result, _cffi_type(379));
}
#else
#  define _cffi_f_TCOD_sys_check_for_event _cffi_d_TCOD_sys_check_for_event
#endif

static char * _cffi_d_TCOD_sys_clipboard_get(void)
{
  return TCOD_sys_clipboard_get();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_clipboard_get(PyObject *self, PyObject *noarg)
{
  char * result;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_sys_clipboard_get(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  return _cffi_from_c_pointer((char *)result, _cffi_type(384));
}
#else
#  define _cffi_f_TCOD_sys_clipboard_get _cffi_d_TCOD_sys_clipboard_get
#endif

static void _cffi_d_TCOD_sys_clipboard_set(char const * x0)
{
  TCOD_sys_clipboard_set(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_clipboard_set(PyObject *self, PyObject *arg0)
{
  char const * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char const *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(64), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_sys_clipboard_set(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_sys_clipboard_set _cffi_d_TCOD_sys_clipboard_set
#endif

static void _cffi_d_TCOD_sys_force_fullscreen_resolution(int x0, int x1)
{
  TCOD_sys_force_fullscreen_resolution(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_force_fullscreen_resolution(PyObject *self, PyObject *args)
{
  int x0;
  int x1;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_sys_force_fullscreen_resolution");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_sys_force_fullscreen_resolution(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_sys_force_fullscreen_resolution _cffi_d_TCOD_sys_force_fullscreen_resolution
#endif

static void _cffi_d_TCOD_sys_get_char_size(int * x0, int * x1)
{
  TCOD_sys_get_char_size(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_get_char_size(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_sys_get_char_size");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(159), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(159), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(159), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(159), arg1) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_sys_get_char_size(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_sys_get_char_size _cffi_d_TCOD_sys_get_char_size
#endif

static void _cffi_d_TCOD_sys_get_current_resolution(int * x0, int * x1)
{
  TCOD_sys_get_current_resolution(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_get_current_resolution(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_sys_get_current_resolution");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(159), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(159), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(159), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(159), arg1) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_sys_get_current_resolution(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_sys_get_current_resolution _cffi_d_TCOD_sys_get_current_resolution
#endif

static int _cffi_d_TCOD_sys_get_fps(void)
{
  return TCOD_sys_get_fps();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_get_fps(PyObject *self, PyObject *noarg)
{
  int result;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_sys_get_fps(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_TCOD_sys_get_fps _cffi_d_TCOD_sys_get_fps
#endif

static void _cffi_d_TCOD_sys_get_fullscreen_offsets(int * x0, int * x1)
{
  TCOD_sys_get_fullscreen_offsets(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_get_fullscreen_offsets(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_sys_get_fullscreen_offsets");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(159), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(159), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(159), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(159), arg1) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_sys_get_fullscreen_offsets(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_sys_get_fullscreen_offsets _cffi_d_TCOD_sys_get_fullscreen_offsets
#endif

static float _cffi_d_TCOD_sys_get_last_frame_length(void)
{
  return TCOD_sys_get_last_frame_length();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_get_last_frame_length(PyObject *self, PyObject *noarg)
{
  float result;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_sys_get_last_frame_length(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  return _cffi_from_c_float(result);
}
#else
#  define _cffi_f_TCOD_sys_get_last_frame_length _cffi_d_TCOD_sys_get_last_frame_length
#endif

static int _cffi_d_TCOD_sys_get_num_cores(void)
{
  return TCOD_sys_get_num_cores();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_get_num_cores(PyObject *self, PyObject *noarg)
{
  int result;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_sys_get_num_cores(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_TCOD_sys_get_num_cores _cffi_d_TCOD_sys_get_num_cores
#endif

static TCOD_renderer_t _cffi_d_TCOD_sys_get_renderer(void)
{
  return TCOD_sys_get_renderer();
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_get_renderer(PyObject *self, PyObject *noarg)
{
  TCOD_renderer_t result;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_sys_get_renderer(); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  (void)noarg; /* unused */
  return _cffi_from_c_deref((char *)&result, _cffi_type(142));
}
#else
#  define _cffi_f_TCOD_sys_get_renderer _cffi_d_TCOD_sys_get_renderer
#endif

static void _cffi_d_TCOD_sys_register_SDL_renderer(void(* x0)(void *))
{
  TCOD_sys_register_SDL_renderer(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_register_SDL_renderer(PyObject *self, PyObject *arg0)
{
  void(* x0)(void *);

  x0 = (void(*)(void *))_cffi_to_c_pointer(arg0, _cffi_type(370));
  if (x0 == (void(*)(void *))NULL && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_sys_register_SDL_renderer(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_sys_register_SDL_renderer _cffi_d_TCOD_sys_register_SDL_renderer
#endif

static void _cffi_d_TCOD_sys_save_screenshot(char const * x0)
{
  TCOD_sys_save_screenshot(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_save_screenshot(PyObject *self, PyObject *arg0)
{
  char const * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char const *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(64), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_sys_save_screenshot(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_sys_save_screenshot _cffi_d_TCOD_sys_save_screenshot
#endif

static void _cffi_d_TCOD_sys_set_fps(int x0)
{
  TCOD_sys_set_fps(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_set_fps(PyObject *self, PyObject *arg0)
{
  int x0;

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_sys_set_fps(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_sys_set_fps _cffi_d_TCOD_sys_set_fps
#endif

static void _cffi_d_TCOD_sys_set_renderer(TCOD_renderer_t x0)
{
  TCOD_sys_set_renderer(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_set_renderer(PyObject *self, PyObject *arg0)
{
  TCOD_renderer_t x0;

  if (_cffi_to_c((char *)&x0, _cffi_type(142), arg0) < 0)
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_sys_set_renderer(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_sys_set_renderer _cffi_d_TCOD_sys_set_renderer
#endif

static void _cffi_d_TCOD_sys_update_char(int x0, int x1, int x2, void * x3, int x4, int x5)
{
  TCOD_sys_update_char(x0, x1, x2, x3, x4, x5);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_update_char(PyObject *self, PyObject *args)
{
  int x0;
  int x1;
  int x2;
  void * x3;
  int x4;
  int x5;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 6, "TCOD_sys_update_char");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (void *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  x4 = _cffi_to_c_int(arg4, int);
  if (x4 == (int)-1 && PyErr_Occurred())
    return NULL;

  x5 = _cffi_to_c_int(arg5, int);
  if (x5 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_sys_update_char(x0, x1, x2, x3, x4, x5); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_sys_update_char _cffi_d_TCOD_sys_update_char
#endif

static TCOD_event_t _cffi_d_TCOD_sys_wait_for_event(int x0, TCOD_key_t * x1, TCOD_mouse_t * x2, unsigned char x3)
{
  return TCOD_sys_wait_for_event(x0, x1, x2, x3);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_sys_wait_for_event(PyObject *self, PyObject *args)
{
  int x0;
  TCOD_key_t * x1;
  TCOD_mouse_t * x2;
  unsigned char x3;
  Py_ssize_t datasize;
  TCOD_event_t result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 4, "TCOD_sys_wait_for_event");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];

  x0 = _cffi_to_c_int(arg0, int);
  if (x0 == (int)-1 && PyErr_Occurred())
    return NULL;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(25), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (TCOD_key_t *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(25), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(26), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (TCOD_mouse_t *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(26), arg2) < 0)
      return NULL;
  }

  x3 = _cffi_to_c_int(arg3, unsigned char);
  if (x3 == (unsigned char)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_sys_wait_for_event(x0, x1, x2, x3); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_deref((char *)&result, _cffi_type(379));
}
#else
#  define _cffi_f_TCOD_sys_wait_for_event _cffi_d_TCOD_sys_wait_for_event
#endif

static void _cffi_d_TCOD_thread_delete(void * x0)
{
  TCOD_thread_delete(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_thread_delete(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_thread_delete(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_thread_delete _cffi_d_TCOD_thread_delete
#endif

static void * _cffi_d_TCOD_thread_new(int(* x0)(void *), void * x1)
{
  return TCOD_thread_new(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_thread_new(PyObject *self, PyObject *args)
{
  int(* x0)(void *);
  void * x1;
  Py_ssize_t datasize;
  void * result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "TCOD_thread_new");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  x0 = (int(*)(void *))_cffi_to_c_pointer(arg0, _cffi_type(117));
  if (x0 == (int(*)(void *))NULL && PyErr_Occurred())
    return NULL;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (void *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = TCOD_thread_new(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_pointer((char *)result, _cffi_type(1));
}
#else
#  define _cffi_f_TCOD_thread_new _cffi_d_TCOD_thread_new
#endif

static void _cffi_d_TCOD_thread_wait(void * x0)
{
  TCOD_thread_wait(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_TCOD_thread_wait(PyObject *self, PyObject *arg0)
{
  void * x0;
  Py_ssize_t datasize;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { TCOD_thread_wait(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_TCOD_thread_wait _cffi_d_TCOD_thread_wait
#endif

static void _cffi_d_set_char(void * x0, int x1, int x2, int x3, int x4, int x5)
{
  set_char(x0, x1, x2, x3, x4, x5);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_set_char(PyObject *self, PyObject *args)
{
  void * x0;
  int x1;
  int x2;
  int x3;
  int x4;
  int x5;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 6, "set_char");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, int);
  if (x1 == (int)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, int);
  if (x3 == (int)-1 && PyErr_Occurred())
    return NULL;

  x4 = _cffi_to_c_int(arg4, int);
  if (x4 == (int)-1 && PyErr_Occurred())
    return NULL;

  x5 = _cffi_to_c_int(arg5, int);
  if (x5 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { set_char(x0, x1, x2, x3, x4, x5); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_set_char _cffi_d_set_char
#endif

static const struct _cffi_global_s _cffi_globals[] = {
  { "TCODK_0", (void *)_cffi_const_TCODK_0, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_1", (void *)_cffi_const_TCODK_1, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_2", (void *)_cffi_const_TCODK_2, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_3", (void *)_cffi_const_TCODK_3, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_4", (void *)_cffi_const_TCODK_4, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_5", (void *)_cffi_const_TCODK_5, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_6", (void *)_cffi_const_TCODK_6, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_7", (void *)_cffi_const_TCODK_7, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_8", (void *)_cffi_const_TCODK_8, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_9", (void *)_cffi_const_TCODK_9, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_ALT", (void *)_cffi_const_TCODK_ALT, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_APPS", (void *)_cffi_const_TCODK_APPS, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_BACKSPACE", (void *)_cffi_const_TCODK_BACKSPACE, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_CAPSLOCK", (void *)_cffi_const_TCODK_CAPSLOCK, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_CHAR", (void *)_cffi_const_TCODK_CHAR, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_CONTROL", (void *)_cffi_const_TCODK_CONTROL, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_DELETE", (void *)_cffi_const_TCODK_DELETE, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_DOWN", (void *)_cffi_const_TCODK_DOWN, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_END", (void *)_cffi_const_TCODK_END, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_ENTER", (void *)_cffi_const_TCODK_ENTER, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_ESCAPE", (void *)_cffi_const_TCODK_ESCAPE, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_F1", (void *)_cffi_const_TCODK_F1, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_F10", (void *)_cffi_const_TCODK_F10, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_F11", (void *)_cffi_const_TCODK_F11, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_F12", (void *)_cffi_const_TCODK_F12, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_F2", (void *)_cffi_const_TCODK_F2, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_F3", (void *)_cffi_const_TCODK_F3, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_F4", (void *)_cffi_const_TCODK_F4, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_F5", (void *)_cffi_const_TCODK_F5, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_F6", (void *)_cffi_const_TCODK_F6, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_F7", (void *)_cffi_const_TCODK_F7, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_F8", (void *)_cffi_const_TCODK_F8, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_F9", (void *)_cffi_const_TCODK_F9, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_HOME", (void *)_cffi_const_TCODK_HOME, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_INSERT", (void *)_cffi_const_TCODK_INSERT, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KP0", (void *)_cffi_const_TCODK_KP0, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KP1", (void *)_cffi_const_TCODK_KP1, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KP2", (void *)_cffi_const_TCODK_KP2, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KP3", (void *)_cffi_const_TCODK_KP3, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KP4", (void *)_cffi_const_TCODK_KP4, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KP5", (void *)_cffi_const_TCODK_KP5, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KP6", (void *)_cffi_const_TCODK_KP6, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KP7", (void *)_cffi_const_TCODK_KP7, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KP8", (void *)_cffi_const_TCODK_KP8, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KP9", (void *)_cffi_const_TCODK_KP9, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KPADD", (void *)_cffi_const_TCODK_KPADD, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KPDEC", (void *)_cffi_const_TCODK_KPDEC, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KPDIV", (void *)_cffi_const_TCODK_KPDIV, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KPENTER", (void *)_cffi_const_TCODK_KPENTER, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KPMUL", (void *)_cffi_const_TCODK_KPMUL, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_KPSUB", (void *)_cffi_const_TCODK_KPSUB, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_LEFT", (void *)_cffi_const_TCODK_LEFT, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_LWIN", (void *)_cffi_const_TCODK_LWIN, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_NONE", (void *)_cffi_const_TCODK_NONE, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_NUMLOCK", (void *)_cffi_const_TCODK_NUMLOCK, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_PAGEDOWN", (void *)_cffi_const_TCODK_PAGEDOWN, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_PAGEUP", (void *)_cffi_const_TCODK_PAGEUP, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_PAUSE", (void *)_cffi_const_TCODK_PAUSE, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_PRINTSCREEN", (void *)_cffi_const_TCODK_PRINTSCREEN, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_RIGHT", (void *)_cffi_const_TCODK_RIGHT, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_RWIN", (void *)_cffi_const_TCODK_RWIN, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_SCROLLLOCK", (void *)_cffi_const_TCODK_SCROLLLOCK, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_SHIFT", (void *)_cffi_const_TCODK_SHIFT, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_SPACE", (void *)_cffi_const_TCODK_SPACE, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_TAB", (void *)_cffi_const_TCODK_TAB, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCODK_UP", (void *)_cffi_const_TCODK_UP, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_BKGND_ADD", (void *)_cffi_const_TCOD_BKGND_ADD, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_BKGND_ADDA", (void *)_cffi_const_TCOD_BKGND_ADDA, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_BKGND_ALPH", (void *)_cffi_const_TCOD_BKGND_ALPH, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_BKGND_BURN", (void *)_cffi_const_TCOD_BKGND_BURN, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_BKGND_COLOR_BURN", (void *)_cffi_const_TCOD_BKGND_COLOR_BURN, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_BKGND_COLOR_DODGE", (void *)_cffi_const_TCOD_BKGND_COLOR_DODGE, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_BKGND_DARKEN", (void *)_cffi_const_TCOD_BKGND_DARKEN, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_BKGND_DEFAULT", (void *)_cffi_const_TCOD_BKGND_DEFAULT, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_BKGND_LIGHTEN", (void *)_cffi_const_TCOD_BKGND_LIGHTEN, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_BKGND_MULTIPLY", (void *)_cffi_const_TCOD_BKGND_MULTIPLY, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_BKGND_NONE", (void *)_cffi_const_TCOD_BKGND_NONE, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_BKGND_OVERLAY", (void *)_cffi_const_TCOD_BKGND_OVERLAY, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_BKGND_SCREEN", (void *)_cffi_const_TCOD_BKGND_SCREEN, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_BKGND_SET", (void *)_cffi_const_TCOD_BKGND_SET, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_CENTER", (void *)_cffi_const_TCOD_CENTER, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_COLCTRL_1", (void *)_cffi_const_TCOD_COLCTRL_1, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_COLCTRL_2", (void *)_cffi_const_TCOD_COLCTRL_2, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_COLCTRL_3", (void *)_cffi_const_TCOD_COLCTRL_3, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_COLCTRL_4", (void *)_cffi_const_TCOD_COLCTRL_4, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_COLCTRL_5", (void *)_cffi_const_TCOD_COLCTRL_5, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_COLCTRL_BACK_RGB", (void *)_cffi_const_TCOD_COLCTRL_BACK_RGB, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_COLCTRL_FORE_RGB", (void *)_cffi_const_TCOD_COLCTRL_FORE_RGB, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_COLCTRL_NUMBER", (void *)_cffi_const_TCOD_COLCTRL_NUMBER, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_COLCTRL_STOP", (void *)_cffi_const_TCOD_COLCTRL_STOP, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_EVENT_ANY", (void *)_cffi_const_TCOD_EVENT_ANY, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_EVENT_KEY", (void *)_cffi_const_TCOD_EVENT_KEY, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_EVENT_KEY_PRESS", (void *)_cffi_const_TCOD_EVENT_KEY_PRESS, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_EVENT_KEY_RELEASE", (void *)_cffi_const_TCOD_EVENT_KEY_RELEASE, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_EVENT_MOUSE", (void *)_cffi_const_TCOD_EVENT_MOUSE, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_EVENT_MOUSE_MOVE", (void *)_cffi_const_TCOD_EVENT_MOUSE_MOVE, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_EVENT_MOUSE_PRESS", (void *)_cffi_const_TCOD_EVENT_MOUSE_PRESS, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_EVENT_MOUSE_RELEASE", (void *)_cffi_const_TCOD_EVENT_MOUSE_RELEASE, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_FONT_LAYOUT_ASCII_INCOL", (void *)_cffi_const_TCOD_FONT_LAYOUT_ASCII_INCOL, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_FONT_LAYOUT_ASCII_INROW", (void *)_cffi_const_TCOD_FONT_LAYOUT_ASCII_INROW, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_FONT_LAYOUT_TCOD", (void *)_cffi_const_TCOD_FONT_LAYOUT_TCOD, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_FONT_TYPE_GRAYSCALE", (void *)_cffi_const_TCOD_FONT_TYPE_GRAYSCALE, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_FONT_TYPE_GREYSCALE", (void *)_cffi_const_TCOD_FONT_TYPE_GREYSCALE, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_KEY_PRESSED", (void *)_cffi_const_TCOD_KEY_PRESSED, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_KEY_RELEASED", (void *)_cffi_const_TCOD_KEY_RELEASED, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_LEFT", (void *)_cffi_const_TCOD_LEFT, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_NB_RENDERERS", (void *)_cffi_const_TCOD_NB_RENDERERS, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_RENDERER_GLSL", (void *)_cffi_const_TCOD_RENDERER_GLSL, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_RENDERER_OPENGL", (void *)_cffi_const_TCOD_RENDERER_OPENGL, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_RENDERER_SDL", (void *)_cffi_const_TCOD_RENDERER_SDL, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_RIGHT", (void *)_cffi_const_TCOD_RIGHT, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "TCOD_close_library", (void *)_cffi_f_TCOD_close_library, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_close_library },
  { "TCOD_condition_broadcast", (void *)_cffi_f_TCOD_condition_broadcast, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_condition_broadcast },
  { "TCOD_condition_delete", (void *)_cffi_f_TCOD_condition_delete, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_condition_delete },
  { "TCOD_condition_new", (void *)_cffi_f_TCOD_condition_new, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 134), (void *)_cffi_d_TCOD_condition_new },
  { "TCOD_condition_signal", (void *)_cffi_f_TCOD_condition_signal, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_condition_signal },
  { "TCOD_condition_wait", (void *)_cffi_f_TCOD_condition_wait, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 336), (void *)_cffi_d_TCOD_condition_wait },
  { "TCOD_console_blit", (void *)_cffi_f_TCOD_console_blit, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 318), (void *)_cffi_d_TCOD_console_blit },
  { "TCOD_console_check_for_keypress", (void *)_cffi_f_TCOD_console_check_for_keypress, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 34), (void *)_cffi_d_TCOD_console_check_for_keypress },
  { "TCOD_console_clear", (void *)_cffi_f_TCOD_console_clear, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_console_clear },
  { "TCOD_console_credits", (void *)_cffi_f_TCOD_console_credits, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 372), (void *)_cffi_d_TCOD_console_credits },
  { "TCOD_console_credits_render", (void *)_cffi_f_TCOD_console_credits_render, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 97), (void *)_cffi_d_TCOD_console_credits_render },
  { "TCOD_console_credits_reset", (void *)_cffi_f_TCOD_console_credits_reset, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 372), (void *)_cffi_d_TCOD_console_credits_reset },
  { "TCOD_console_delete", (void *)_cffi_f_TCOD_console_delete, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_console_delete },
  { "TCOD_console_disable_keyboard_repeat", (void *)_cffi_f_TCOD_console_disable_keyboard_repeat, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 372), (void *)_cffi_d_TCOD_console_disable_keyboard_repeat },
  { "TCOD_console_flush", (void *)_cffi_f_TCOD_console_flush, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 372), (void *)_cffi_d_TCOD_console_flush },
  { "TCOD_console_from_file", (void *)_cffi_f_TCOD_console_from_file, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 113), (void *)_cffi_d_TCOD_console_from_file },
  { "TCOD_console_get_alignment", (void *)_cffi_f_TCOD_console_get_alignment, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 0), (void *)_cffi_d_TCOD_console_get_alignment },
  { "TCOD_console_get_background_flag", (void *)_cffi_f_TCOD_console_get_background_flag, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 3), (void *)_cffi_d_TCOD_console_get_background_flag },
  { "TCOD_console_get_char", (void *)_cffi_f_TCOD_console_get_char, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 51), (void *)_cffi_d_TCOD_console_get_char },
  { "TCOD_console_get_char_background", (void *)_cffi_f_TCOD_console_get_char_background, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 16), (void *)_cffi_d_TCOD_console_get_char_background },
  { "TCOD_console_get_char_foreground", (void *)_cffi_f_TCOD_console_get_char_foreground, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 16), (void *)_cffi_d_TCOD_console_get_char_foreground },
  { "TCOD_console_get_default_background", (void *)_cffi_f_TCOD_console_get_default_background, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 6), (void *)_cffi_d_TCOD_console_get_default_background },
  { "TCOD_console_get_default_foreground", (void *)_cffi_f_TCOD_console_get_default_foreground, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 6), (void *)_cffi_d_TCOD_console_get_default_foreground },
  { "TCOD_console_get_fade", (void *)_cffi_f_TCOD_console_get_fade, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 111), (void *)_cffi_d_TCOD_console_get_fade },
  { "TCOD_console_get_fading_color", (void *)_cffi_f_TCOD_console_get_fading_color, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 21), (void *)_cffi_d_TCOD_console_get_fading_color },
  { "TCOD_console_get_height", (void *)_cffi_f_TCOD_console_get_height, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 48), (void *)_cffi_d_TCOD_console_get_height },
  { "TCOD_console_get_height_rect", (void *)_cffi_const_TCOD_console_get_height_rect, _CFFI_OP(_CFFI_OP_CONSTANT, 388), (void *)0 },
  { "TCOD_console_get_height_rect_utf", (void *)_cffi_const_TCOD_console_get_height_rect_utf, _CFFI_OP(_CFFI_OP_CONSTANT, 389), (void *)0 },
  { "TCOD_console_get_width", (void *)_cffi_f_TCOD_console_get_width, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 48), (void *)_cffi_d_TCOD_console_get_width },
  { "TCOD_console_hline", (void *)_cffi_f_TCOD_console_hline, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 276), (void *)_cffi_d_TCOD_console_hline },
  { "TCOD_console_init_root", (void *)_cffi_f_TCOD_console_init_root, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 169), (void *)_cffi_d_TCOD_console_init_root },
  { "TCOD_console_is_fullscreen", (void *)_cffi_f_TCOD_console_is_fullscreen, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 111), (void *)_cffi_d_TCOD_console_is_fullscreen },
  { "TCOD_console_is_key_pressed", (void *)_cffi_f_TCOD_console_is_key_pressed, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 94), (void *)_cffi_d_TCOD_console_is_key_pressed },
  { "TCOD_console_is_window_closed", (void *)_cffi_f_TCOD_console_is_window_closed, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 111), (void *)_cffi_d_TCOD_console_is_window_closed },
  { "TCOD_console_load_apf", (void *)_cffi_f_TCOD_console_load_apf, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 102), (void *)_cffi_d_TCOD_console_load_apf },
  { "TCOD_console_load_asc", (void *)_cffi_f_TCOD_console_load_asc, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 102), (void *)_cffi_d_TCOD_console_load_asc },
  { "TCOD_console_map_ascii_code_to_font", (void *)_cffi_f_TCOD_console_map_ascii_code_to_font, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 176), (void *)_cffi_d_TCOD_console_map_ascii_code_to_font },
  { "TCOD_console_map_ascii_codes_to_font", (void *)_cffi_f_TCOD_console_map_ascii_codes_to_font, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 181), (void *)_cffi_d_TCOD_console_map_ascii_codes_to_font },
  { "TCOD_console_map_string_to_font", (void *)_cffi_f_TCOD_console_map_string_to_font, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 147), (void *)_cffi_d_TCOD_console_map_string_to_font },
  { "TCOD_console_map_string_to_font_utf", (void *)_cffi_f_TCOD_console_map_string_to_font_utf, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 374), (void *)_cffi_d_TCOD_console_map_string_to_font_utf },
  { "TCOD_console_new", (void *)_cffi_f_TCOD_console_new, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 123), (void *)_cffi_d_TCOD_console_new },
  { "TCOD_console_print", (void *)_cffi_const_TCOD_console_print, _CFFI_OP(_CFFI_OP_CONSTANT, 397), (void *)0 },
  { "TCOD_console_print_ex", (void *)_cffi_const_TCOD_console_print_ex, _CFFI_OP(_CFFI_OP_CONSTANT, 395), (void *)0 },
  { "TCOD_console_print_ex_utf", (void *)_cffi_const_TCOD_console_print_ex_utf, _CFFI_OP(_CFFI_OP_CONSTANT, 396), (void *)0 },
  { "TCOD_console_print_frame", (void *)_cffi_const_TCOD_console_print_frame, _CFFI_OP(_CFFI_OP_CONSTANT, 398), (void *)0 },
  { "TCOD_console_print_rect", (void *)_cffi_const_TCOD_console_print_rect, _CFFI_OP(_CFFI_OP_CONSTANT, 388), (void *)0 },
  { "TCOD_console_print_rect_ex", (void *)_cffi_const_TCOD_console_print_rect_ex, _CFFI_OP(_CFFI_OP_CONSTANT, 386), (void *)0 },
  { "TCOD_console_print_rect_ex_utf", (void *)_cffi_const_TCOD_console_print_rect_ex_utf, _CFFI_OP(_CFFI_OP_CONSTANT, 387), (void *)0 },
  { "TCOD_console_print_rect_utf", (void *)_cffi_const_TCOD_console_print_rect_utf, _CFFI_OP(_CFFI_OP_CONSTANT, 389), (void *)0 },
  { "TCOD_console_print_utf", (void *)_cffi_const_TCOD_console_print_utf, _CFFI_OP(_CFFI_OP_CONSTANT, 399), (void *)0 },
  { "TCOD_console_put_char", (void *)_cffi_f_TCOD_console_put_char, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 276), (void *)_cffi_d_TCOD_console_put_char },
  { "TCOD_console_put_char_ex", (void *)_cffi_f_TCOD_console_put_char_ex, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 283), (void *)_cffi_d_TCOD_console_put_char_ex },
  { "TCOD_console_rect", (void *)_cffi_f_TCOD_console_rect, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 299), (void *)_cffi_d_TCOD_console_rect },
  { "TCOD_console_save_apf", (void *)_cffi_f_TCOD_console_save_apf, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 102), (void *)_cffi_d_TCOD_console_save_apf },
  { "TCOD_console_save_asc", (void *)_cffi_f_TCOD_console_save_asc, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 102), (void *)_cffi_d_TCOD_console_save_asc },
  { "TCOD_console_set_alignment", (void *)_cffi_f_TCOD_console_set_alignment, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 205), (void *)_cffi_d_TCOD_console_set_alignment },
  { "TCOD_console_set_background_flag", (void *)_cffi_f_TCOD_console_set_background_flag, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 209), (void *)_cffi_d_TCOD_console_set_background_flag },
  { "TCOD_console_set_char", (void *)_cffi_f_TCOD_console_set_char, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 270), (void *)_cffi_d_TCOD_console_set_char },
  { "TCOD_console_set_char_background", (void *)_cffi_f_TCOD_console_set_char_background, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 257), (void *)_cffi_d_TCOD_console_set_char_background },
  { "TCOD_console_set_char_foreground", (void *)_cffi_f_TCOD_console_set_char_foreground, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 251), (void *)_cffi_d_TCOD_console_set_char_foreground },
  { "TCOD_console_set_color_control", (void *)_cffi_f_TCOD_console_set_color_control, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 136), (void *)_cffi_d_TCOD_console_set_color_control },
  { "TCOD_console_set_custom_font", (void *)_cffi_f_TCOD_console_set_custom_font, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 152), (void *)_cffi_d_TCOD_console_set_custom_font },
  { "TCOD_console_set_default_background", (void *)_cffi_f_TCOD_console_set_default_background, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 213), (void *)_cffi_d_TCOD_console_set_default_background },
  { "TCOD_console_set_default_foreground", (void *)_cffi_f_TCOD_console_set_default_foreground, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 213), (void *)_cffi_d_TCOD_console_set_default_foreground },
  { "TCOD_console_set_dirty", (void *)_cffi_f_TCOD_console_set_dirty, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 181), (void *)_cffi_d_TCOD_console_set_dirty },
  { "TCOD_console_set_fade", (void *)_cffi_f_TCOD_console_set_fade, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 198), (void *)_cffi_d_TCOD_console_set_fade },
  { "TCOD_console_set_fullscreen", (void *)_cffi_f_TCOD_console_set_fullscreen, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 195), (void *)_cffi_d_TCOD_console_set_fullscreen },
  { "TCOD_console_set_key_color", (void *)_cffi_f_TCOD_console_set_key_color, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 213), (void *)_cffi_d_TCOD_console_set_key_color },
  { "TCOD_console_set_keyboard_repeat", (void *)_cffi_f_TCOD_console_set_keyboard_repeat, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 165), (void *)_cffi_d_TCOD_console_set_keyboard_repeat },
  { "TCOD_console_set_window_title", (void *)_cffi_f_TCOD_console_set_window_title, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 144), (void *)_cffi_d_TCOD_console_set_window_title },
  { "TCOD_console_vline", (void *)_cffi_f_TCOD_console_vline, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 276), (void *)_cffi_d_TCOD_console_vline },
  { "TCOD_console_wait_for_keypress", (void *)_cffi_f_TCOD_console_wait_for_keypress, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 37), (void *)_cffi_d_TCOD_console_wait_for_keypress },
  { "TCOD_get_function_address", (void *)_cffi_f_TCOD_get_function_address, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 130), (void *)_cffi_d_TCOD_get_function_address },
  { "TCOD_image_blit", (void *)_cffi_f_TCOD_image_blit, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 340), (void *)_cffi_d_TCOD_image_blit },
  { "TCOD_image_blit_2x", (void *)_cffi_f_TCOD_image_blit_2x, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 359), (void *)_cffi_d_TCOD_image_blit_2x },
  { "TCOD_image_blit_rect", (void *)_cffi_f_TCOD_image_blit_rect, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 350), (void *)_cffi_d_TCOD_image_blit_rect },
  { "TCOD_image_clear", (void *)_cffi_f_TCOD_image_clear, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 213), (void *)_cffi_d_TCOD_image_clear },
  { "TCOD_image_delete", (void *)_cffi_f_TCOD_image_delete, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_image_delete },
  { "TCOD_image_from_console", (void *)_cffi_f_TCOD_image_from_console, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 127), (void *)_cffi_d_TCOD_image_from_console },
  { "TCOD_image_get_alpha", (void *)_cffi_f_TCOD_image_get_alpha, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 51), (void *)_cffi_d_TCOD_image_get_alpha },
  { "TCOD_image_get_mipmap_pixel", (void *)_cffi_f_TCOD_image_get_mipmap_pixel, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 9), (void *)_cffi_d_TCOD_image_get_mipmap_pixel },
  { "TCOD_image_get_pixel", (void *)_cffi_f_TCOD_image_get_pixel, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 16), (void *)_cffi_d_TCOD_image_get_pixel },
  { "TCOD_image_get_size", (void *)_cffi_f_TCOD_image_get_size, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 221), (void *)_cffi_d_TCOD_image_get_size },
  { "TCOD_image_hflip", (void *)_cffi_f_TCOD_image_hflip, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_image_hflip },
  { "TCOD_image_invert", (void *)_cffi_f_TCOD_image_invert, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_image_invert },
  { "TCOD_image_is_pixel_transparent", (void *)_cffi_f_TCOD_image_is_pixel_transparent, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 106), (void *)_cffi_d_TCOD_image_is_pixel_transparent },
  { "TCOD_image_load", (void *)_cffi_f_TCOD_image_load, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 113), (void *)_cffi_d_TCOD_image_load },
  { "TCOD_image_new", (void *)_cffi_f_TCOD_image_new, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 123), (void *)_cffi_d_TCOD_image_new },
  { "TCOD_image_put_pixel", (void *)_cffi_f_TCOD_image_put_pixel, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 251), (void *)_cffi_d_TCOD_image_put_pixel },
  { "TCOD_image_refresh_console", (void *)_cffi_f_TCOD_image_refresh_console, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 336), (void *)_cffi_d_TCOD_image_refresh_console },
  { "TCOD_image_rotate90", (void *)_cffi_f_TCOD_image_rotate90, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 226), (void *)_cffi_d_TCOD_image_rotate90 },
  { "TCOD_image_save", (void *)_cffi_f_TCOD_image_save, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 217), (void *)_cffi_d_TCOD_image_save },
  { "TCOD_image_scale", (void *)_cffi_f_TCOD_image_scale, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 230), (void *)_cffi_d_TCOD_image_scale },
  { "TCOD_image_set_key_color", (void *)_cffi_f_TCOD_image_set_key_color, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 213), (void *)_cffi_d_TCOD_image_set_key_color },
  { "TCOD_image_vflip", (void *)_cffi_f_TCOD_image_vflip, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_image_vflip },
  { "TCOD_load_library", (void *)_cffi_f_TCOD_load_library, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 113), (void *)_cffi_d_TCOD_load_library },
  { "TCOD_mouse_get_status", (void *)_cffi_f_TCOD_mouse_get_status, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 40), (void *)_cffi_d_TCOD_mouse_get_status },
  { "TCOD_mouse_is_cursor_visible", (void *)_cffi_f_TCOD_mouse_is_cursor_visible, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 111), (void *)_cffi_d_TCOD_mouse_is_cursor_visible },
  { "TCOD_mouse_move", (void *)_cffi_f_TCOD_mouse_move, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 165), (void *)_cffi_d_TCOD_mouse_move },
  { "TCOD_mouse_show_cursor", (void *)_cffi_f_TCOD_mouse_show_cursor, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 195), (void *)_cffi_d_TCOD_mouse_show_cursor },
  { "TCOD_mutex_delete", (void *)_cffi_f_TCOD_mutex_delete, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_mutex_delete },
  { "TCOD_mutex_in", (void *)_cffi_f_TCOD_mutex_in, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_mutex_in },
  { "TCOD_mutex_new", (void *)_cffi_f_TCOD_mutex_new, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 134), (void *)_cffi_d_TCOD_mutex_new },
  { "TCOD_mutex_out", (void *)_cffi_f_TCOD_mutex_out, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_mutex_out },
  { "TCOD_semaphore_delete", (void *)_cffi_f_TCOD_semaphore_delete, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_semaphore_delete },
  { "TCOD_semaphore_lock", (void *)_cffi_f_TCOD_semaphore_lock, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_semaphore_lock },
  { "TCOD_semaphore_new", (void *)_cffi_f_TCOD_semaphore_new, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 120), (void *)_cffi_d_TCOD_semaphore_new },
  { "TCOD_semaphore_unlock", (void *)_cffi_f_TCOD_semaphore_unlock, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_semaphore_unlock },
  { "TCOD_sys_check_for_event", (void *)_cffi_f_TCOD_sys_check_for_event, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 23), (void *)_cffi_d_TCOD_sys_check_for_event },
  { "TCOD_sys_clipboard_get", (void *)_cffi_f_TCOD_sys_clipboard_get, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 44), (void *)_cffi_d_TCOD_sys_clipboard_get },
  { "TCOD_sys_clipboard_set", (void *)_cffi_f_TCOD_sys_clipboard_set, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 144), (void *)_cffi_d_TCOD_sys_clipboard_set },
  { "TCOD_sys_force_fullscreen_resolution", (void *)_cffi_f_TCOD_sys_force_fullscreen_resolution, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 165), (void *)_cffi_d_TCOD_sys_force_fullscreen_resolution },
  { "TCOD_sys_get_char_size", (void *)_cffi_f_TCOD_sys_get_char_size, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 158), (void *)_cffi_d_TCOD_sys_get_char_size },
  { "TCOD_sys_get_current_resolution", (void *)_cffi_f_TCOD_sys_get_current_resolution, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 158), (void *)_cffi_d_TCOD_sys_get_current_resolution },
  { "TCOD_sys_get_fps", (void *)_cffi_f_TCOD_sys_get_fps, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 92), (void *)_cffi_d_TCOD_sys_get_fps },
  { "TCOD_sys_get_fullscreen_offsets", (void *)_cffi_f_TCOD_sys_get_fullscreen_offsets, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 158), (void *)_cffi_d_TCOD_sys_get_fullscreen_offsets },
  { "TCOD_sys_get_last_frame_length", (void *)_cffi_f_TCOD_sys_get_last_frame_length, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 46), (void *)_cffi_d_TCOD_sys_get_last_frame_length },
  { "TCOD_sys_get_num_cores", (void *)_cffi_f_TCOD_sys_get_num_cores, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 92), (void *)_cffi_d_TCOD_sys_get_num_cores },
  { "TCOD_sys_get_renderer", (void *)_cffi_f_TCOD_sys_get_renderer, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_N, 42), (void *)_cffi_d_TCOD_sys_get_renderer },
  { "TCOD_sys_register_SDL_renderer", (void *)_cffi_f_TCOD_sys_register_SDL_renderer, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 369), (void *)_cffi_d_TCOD_sys_register_SDL_renderer },
  { "TCOD_sys_save_screenshot", (void *)_cffi_f_TCOD_sys_save_screenshot, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 144), (void *)_cffi_d_TCOD_sys_save_screenshot },
  { "TCOD_sys_set_fps", (void *)_cffi_f_TCOD_sys_set_fps, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 162), (void *)_cffi_d_TCOD_sys_set_fps },
  { "TCOD_sys_set_renderer", (void *)_cffi_f_TCOD_sys_set_renderer, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 141), (void *)_cffi_d_TCOD_sys_set_renderer },
  { "TCOD_sys_update_char", (void *)_cffi_f_TCOD_sys_update_char, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 187), (void *)_cffi_d_TCOD_sys_update_char },
  { "TCOD_sys_wait_for_event", (void *)_cffi_f_TCOD_sys_wait_for_event, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 28), (void *)_cffi_d_TCOD_sys_wait_for_event },
  { "TCOD_thread_delete", (void *)_cffi_f_TCOD_thread_delete, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_thread_delete },
  { "TCOD_thread_new", (void *)_cffi_f_TCOD_thread_new, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 116), (void *)_cffi_d_TCOD_thread_new },
  { "TCOD_thread_wait", (void *)_cffi_f_TCOD_thread_wait, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 202), (void *)_cffi_d_TCOD_thread_wait },
  { "set_char", (void *)_cffi_f_set_char, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 291), (void *)_cffi_d_set_char },
};

static const struct _cffi_field_s _cffi_fields[] = {
  { "r", offsetof(TCOD_color_t, r),
         sizeof(((TCOD_color_t *)0)->r),
         _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "g", offsetof(TCOD_color_t, g),
         sizeof(((TCOD_color_t *)0)->g),
         _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "b", offsetof(TCOD_color_t, b),
         sizeof(((TCOD_color_t *)0)->b),
         _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "vk", offsetof(TCOD_key_t, vk),
          sizeof(((TCOD_key_t *)0)->vk),
          _CFFI_OP(_CFFI_OP_NOOP, 95) },
  { "c", offsetof(TCOD_key_t, c),
         sizeof(((TCOD_key_t *)0)->c),
         _CFFI_OP(_CFFI_OP_NOOP, 385) },
  { "pressed", offsetof(TCOD_key_t, pressed),
               sizeof(((TCOD_key_t *)0)->pressed),
               _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "lalt", offsetof(TCOD_key_t, lalt),
            sizeof(((TCOD_key_t *)0)->lalt),
            _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "lctrl", offsetof(TCOD_key_t, lctrl),
             sizeof(((TCOD_key_t *)0)->lctrl),
             _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "ralt", offsetof(TCOD_key_t, ralt),
            sizeof(((TCOD_key_t *)0)->ralt),
            _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "rctrl", offsetof(TCOD_key_t, rctrl),
             sizeof(((TCOD_key_t *)0)->rctrl),
             _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "shift", offsetof(TCOD_key_t, shift),
             sizeof(((TCOD_key_t *)0)->shift),
             _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "x", offsetof(TCOD_mouse_t, x),
         sizeof(((TCOD_mouse_t *)0)->x),
         _CFFI_OP(_CFFI_OP_NOOP, 18) },
  { "y", offsetof(TCOD_mouse_t, y),
         sizeof(((TCOD_mouse_t *)0)->y),
         _CFFI_OP(_CFFI_OP_NOOP, 18) },
  { "dx", offsetof(TCOD_mouse_t, dx),
          sizeof(((TCOD_mouse_t *)0)->dx),
          _CFFI_OP(_CFFI_OP_NOOP, 18) },
  { "dy", offsetof(TCOD_mouse_t, dy),
          sizeof(((TCOD_mouse_t *)0)->dy),
          _CFFI_OP(_CFFI_OP_NOOP, 18) },
  { "cx", offsetof(TCOD_mouse_t, cx),
          sizeof(((TCOD_mouse_t *)0)->cx),
          _CFFI_OP(_CFFI_OP_NOOP, 18) },
  { "cy", offsetof(TCOD_mouse_t, cy),
          sizeof(((TCOD_mouse_t *)0)->cy),
          _CFFI_OP(_CFFI_OP_NOOP, 18) },
  { "dcx", offsetof(TCOD_mouse_t, dcx),
           sizeof(((TCOD_mouse_t *)0)->dcx),
           _CFFI_OP(_CFFI_OP_NOOP, 18) },
  { "dcy", offsetof(TCOD_mouse_t, dcy),
           sizeof(((TCOD_mouse_t *)0)->dcy),
           _CFFI_OP(_CFFI_OP_NOOP, 18) },
  { "lbutton", offsetof(TCOD_mouse_t, lbutton),
               sizeof(((TCOD_mouse_t *)0)->lbutton),
               _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "rbutton", offsetof(TCOD_mouse_t, rbutton),
               sizeof(((TCOD_mouse_t *)0)->rbutton),
               _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "mbutton", offsetof(TCOD_mouse_t, mbutton),
               sizeof(((TCOD_mouse_t *)0)->mbutton),
               _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "lbutton_pressed", offsetof(TCOD_mouse_t, lbutton_pressed),
                       sizeof(((TCOD_mouse_t *)0)->lbutton_pressed),
                       _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "rbutton_pressed", offsetof(TCOD_mouse_t, rbutton_pressed),
                       sizeof(((TCOD_mouse_t *)0)->rbutton_pressed),
                       _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "mbutton_pressed", offsetof(TCOD_mouse_t, mbutton_pressed),
                       sizeof(((TCOD_mouse_t *)0)->mbutton_pressed),
                       _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "wheel_up", offsetof(TCOD_mouse_t, wheel_up),
                sizeof(((TCOD_mouse_t *)0)->wheel_up),
                _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "wheel_down", offsetof(TCOD_mouse_t, wheel_down),
                  sizeof(((TCOD_mouse_t *)0)->wheel_down),
                  _CFFI_OP(_CFFI_OP_NOOP, 32) },
};

static const struct _cffi_struct_union_s _cffi_struct_unions[] = {
  { "$TCOD_color_t", 138, _CFFI_F_CHECK_FIELDS,
    sizeof(TCOD_color_t), offsetof(struct _cffi_align_typedef_TCOD_color_t, y), 0, 3 },
  { "$TCOD_key_t", 382, _CFFI_F_CHECK_FIELDS,
    sizeof(TCOD_key_t), offsetof(struct _cffi_align_typedef_TCOD_key_t, y), 3, 8 },
  { "$TCOD_mouse_t", 383, _CFFI_F_CHECK_FIELDS,
    sizeof(TCOD_mouse_t), offsetof(struct _cffi_align_typedef_TCOD_mouse_t, y), 11, 16 },
};

static const struct _cffi_enum_s _cffi_enums[] = {
  { "$TCOD_alignment_t", 63, _cffi_prim_int(sizeof(TCOD_alignment_t), ((TCOD_alignment_t)-1) <= 0),
    "TCOD_LEFT,TCOD_RIGHT,TCOD_CENTER" },
  { "$TCOD_bkgnd_flag_t", 62, _cffi_prim_int(sizeof(TCOD_bkgnd_flag_t), ((TCOD_bkgnd_flag_t)-1) <= 0),
    "TCOD_BKGND_NONE,TCOD_BKGND_SET,TCOD_BKGND_MULTIPLY,TCOD_BKGND_LIGHTEN,TCOD_BKGND_DARKEN,TCOD_BKGND_SCREEN,TCOD_BKGND_COLOR_DODGE,TCOD_BKGND_COLOR_BURN,TCOD_BKGND_ADD,TCOD_BKGND_ADDA,TCOD_BKGND_BURN,TCOD_BKGND_OVERLAY,TCOD_BKGND_ALPH,TCOD_BKGND_DEFAULT" },
  { "$TCOD_colctrl_t", 137, _cffi_prim_int(sizeof(TCOD_colctrl_t), ((TCOD_colctrl_t)-1) <= 0),
    "TCOD_COLCTRL_1,TCOD_COLCTRL_2,TCOD_COLCTRL_3,TCOD_COLCTRL_4,TCOD_COLCTRL_5,TCOD_COLCTRL_NUMBER,TCOD_COLCTRL_FORE_RGB,TCOD_COLCTRL_BACK_RGB,TCOD_COLCTRL_STOP" },
  { "$TCOD_event_t", 379, _cffi_prim_int(sizeof(TCOD_event_t), ((TCOD_event_t)-1) <= 0),
    "TCOD_EVENT_KEY_PRESS,TCOD_EVENT_KEY_RELEASE,TCOD_EVENT_KEY,TCOD_EVENT_MOUSE_MOVE,TCOD_EVENT_MOUSE_PRESS,TCOD_EVENT_MOUSE_RELEASE,TCOD_EVENT_MOUSE,TCOD_EVENT_ANY" },
  { "$TCOD_font_flags_t", 380, _cffi_prim_int(sizeof(TCOD_font_flags_t), ((TCOD_font_flags_t)-1) <= 0),
    "TCOD_FONT_LAYOUT_ASCII_INCOL,TCOD_FONT_LAYOUT_ASCII_INROW,TCOD_FONT_TYPE_GREYSCALE,TCOD_FONT_TYPE_GRAYSCALE,TCOD_FONT_LAYOUT_TCOD" },
  { "$TCOD_key_status_t", 381, _cffi_prim_int(sizeof(TCOD_key_status_t), ((TCOD_key_status_t)-1) <= 0),
    "TCOD_KEY_PRESSED,TCOD_KEY_RELEASED" },
  { "$TCOD_keycode_t", 95, _cffi_prim_int(sizeof(TCOD_keycode_t), ((TCOD_keycode_t)-1) <= 0),
    "TCODK_NONE,TCODK_ESCAPE,TCODK_BACKSPACE,TCODK_TAB,TCODK_ENTER,TCODK_SHIFT,TCODK_CONTROL,TCODK_ALT,TCODK_PAUSE,TCODK_CAPSLOCK,TCODK_PAGEUP,TCODK_PAGEDOWN,TCODK_END,TCODK_HOME,TCODK_UP,TCODK_LEFT,TCODK_RIGHT,TCODK_DOWN,TCODK_PRINTSCREEN,TCODK_INSERT,TCODK_DELETE,TCODK_LWIN,TCODK_RWIN,TCODK_APPS,TCODK_0,TCODK_1,TCODK_2,TCODK_3,TCODK_4,TCODK_5,TCODK_6,TCODK_7,TCODK_8,TCODK_9,TCODK_KP0,TCODK_KP1,TCODK_KP2,TCODK_KP3,TCODK_KP4,TCODK_KP5,TCODK_KP6,TCODK_KP7,TCODK_KP8,TCODK_KP9,TCODK_KPADD,TCODK_KPSUB,TCODK_KPDIV,TCODK_KPMUL,TCODK_KPDEC,TCODK_KPENTER,TCODK_F1,TCODK_F2,TCODK_F3,TCODK_F4,TCODK_F5,TCODK_F6,TCODK_F7,TCODK_F8,TCODK_F9,TCODK_F10,TCODK_F11,TCODK_F12,TCODK_NUMLOCK,TCODK_SCROLLLOCK,TCODK_SPACE,TCODK_CHAR" },
  { "$TCOD_renderer_t", 142, _cffi_prim_int(sizeof(TCOD_renderer_t), ((TCOD_renderer_t)-1) <= 0),
    "TCOD_RENDERER_GLSL,TCOD_RENDERER_OPENGL,TCOD_RENDERER_SDL,TCOD_NB_RENDERERS" },
};

static const struct _cffi_typename_s _cffi_typenames[] = {
  { "SDL_renderer_t", 370 },
  { "TCOD_alignment_t", 63 },
  { "TCOD_bkgnd_flag_t", 62 },
  { "TCOD_colctrl_t", 137 },
  { "TCOD_color_t", 138 },
  { "TCOD_cond_t", 1 },
  { "TCOD_console_t", 1 },
  { "TCOD_event_t", 379 },
  { "TCOD_font_flags_t", 380 },
  { "TCOD_image_t", 1 },
  { "TCOD_key_status_t", 381 },
  { "TCOD_key_t", 382 },
  { "TCOD_keycode_t", 95 },
  { "TCOD_library_t", 1 },
  { "TCOD_mouse_t", 383 },
  { "TCOD_mutex_t", 1 },
  { "TCOD_renderer_t", 142 },
  { "TCOD_semaphore_t", 1 },
  { "TCOD_thread_t", 1 },
  { "bool", 32 },
  { "int16", 391 },
  { "int32", 18 },
  { "int8", 385 },
  { "intptr", 390 },
  { "uint16", 394 },
  { "uint32", 392 },
  { "uint8", 32 },
  { "uintptr", 393 },
};

static const struct _cffi_type_context_s _cffi_type_context = {
  _cffi_types,
  _cffi_globals,
  _cffi_fields,
  _cffi_struct_unions,
  _cffi_enums,
  _cffi_typenames,
  240,  /* num_globals */
  3,  /* num_struct_unions */
  8,  /* num_enums */
  28,  /* num_typenames */
  NULL,  /* no includes */
  402,  /* num_types */
  0,  /* flags */
};

#ifdef PYPY_VERSION
PyMODINIT_FUNC
_cffi_pypyinit__libtcod(const void *p[])
{
    p[0] = (const void *)0x2601;
    p[1] = &_cffi_type_context;
}
#  ifdef _MSC_VER
     PyMODINIT_FUNC
#  if PY_MAJOR_VERSION >= 3
     PyInit__libtcod(void) { return NULL; }
#  else
     init_libtcod(void) { }
#  endif
#  endif
#elif PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit__libtcod(void)
{
  return _cffi_init("_libtcod", 0x2601, &_cffi_type_context);
}
#else
PyMODINIT_FUNC
init_libtcod(void)
{
  _cffi_init("_libtcod", 0x2601, &_cffi_type_context);
}
#endif
