/*
  pygame - Python Game Library
  Copyright (C) 2009 Vicent Marti

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public
  License along with this library; if not, write to the Free
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*/

/* Pythons encoding is not quite what we want. Python UTF-16 and 32 encodings
 * pass through surrogate area codes untouched, and unchecked. So if one must
 * do ones own UTF-16 surrogate pair checking then one might as well perform
 * the entire translation.
 */

#define PYGAME_FREETYPE_INTERNAL
#define NO_PYGAME_C_API

#include "ft_wrap.h"

#define SIZEOF_PGFT_STRING(len) \
    (sizeof(PGFT_String) + (Py_ssize_t)(len) * sizeof(PGFT_char))

static const PGFT_char UNICODE_HSA_START = 0xD800;
static const PGFT_char UNICODE_HSA_END = 0xDBFF;
static const PGFT_char UNICODE_LSA_START = 0xDC00;
static const PGFT_char UNICODE_LSA_END = 0xDFFF;
static const PGFT_char UNICODE_SA_START = 0xD800;
static const PGFT_char UNICODE_SA_END = 0xDFFF;

static void
raise_unicode_error(const char *codec, PyObject *unistr, Py_ssize_t start,
                    Py_ssize_t end, const char *reason)
{
    PyObject *e = PyObject_CallFunction(PyExc_UnicodeEncodeError, "sSkks",
                                        codec, unistr, (unsigned long)start,
                                        (unsigned long)end, reason);

    if (!e)
        return;

    Py_INCREF(PyExc_UnicodeEncodeError);
    PyErr_Restore(PyExc_UnicodeEncodeError, e, 0);
}

/* Helper for _PGFT_EncodePyString to handle PyUnicode object */
static PGFT_String *
_encode_unicode_string(PyObject *obj, int ucs4)
{
    PGFT_String *utf32_buffer = NULL;
    PGFT_char *dst;
    Py_ssize_t len, srclen;
    PGFT_char c;
    int i, j;
    /* This Py_UCS4 src has to be freed later */
    Py_UCS4 *src = PyUnicode_AsUCS4Copy(obj);
    if (!src)
        return NULL;
    len = srclen = PyUnicode_GetLength(obj);

    if (!ucs4) {
        /* Do UTF-16 surrogate pair decoding. Calculate character count
         * and raise an exception on a malformed surrogate pair.
         */
        for (i = 0; i < srclen; ++i) {
            c = (PGFT_char)src[i];
            if (c >= UNICODE_SA_START && c <= UNICODE_SA_END) {
                if (c > UNICODE_HSA_END) {
                    raise_unicode_error("utf-32", obj, i, i + 1,
                                        "missing high-surrogate code point");
                    goto end;
                }
                if (++i == srclen) {
                    raise_unicode_error("utf-32", obj, i - 1, i,
                                        "missing low-surrogate code point");
                    goto end;
                }
                c = (PGFT_char)src[i];
                if (c < UNICODE_LSA_START || c > UNICODE_LSA_END) {
                    raise_unicode_error("utf-32", obj, i, i + 1,
                                        "expected low-surrogate code point");
                    goto end;
                }
                --len;
            }
        }
    }

    utf32_buffer = (PGFT_String *)_PGFT_malloc(SIZEOF_PGFT_STRING(len));
    if (!utf32_buffer) {
        PyErr_NoMemory();
        goto end;
    }
    dst = utf32_buffer->data;
    if (!ucs4) {
        for (i = 0, j = 0; i < srclen; ++i, ++j) {
            c = (PGFT_char)src[i];
            if (c >= UNICODE_HSA_START && c <= UNICODE_HSA_END) {
                c = ((c & 0x3FF) << 10 | (PGFT_char)(src[++i] & 0x3FF)) +
                    0x10000U;
            }
            dst[j] = c;
        }
    }
    else {
        for (i = 0; i < srclen; ++i) {
            dst[i] = (PGFT_char)src[i];
        }
    }

end:
    PyMem_Free(src);
    if (utf32_buffer) {
        utf32_buffer->data[len] = 0;
        utf32_buffer->length = len;
    }
    return utf32_buffer;
}

/* Helper for _PGFT_EncodePyString to handle Bytes object */
static PGFT_String *
_encode_bytes_string(PyObject *obj)
{
    /*
     * For bytes objects, assume the bytes are
     * Latin1 text (who would manually enter bytes as
     * UTF8 anyway?), so manually copy the raw contents
     * of the object expanding each byte to 32 bits.
     */
    PGFT_String *utf32_buffer;
    PGFT_char *dst;
    Py_ssize_t len;
    int i;
    char *src;

    PyBytes_AsStringAndSize(obj, &src, &len);
    utf32_buffer = (PGFT_String *)_PGFT_malloc(SIZEOF_PGFT_STRING(len));
    if (!utf32_buffer) {
        PyErr_NoMemory();
        return NULL;
    }
    dst = utf32_buffer->data;
    for (i = 0; i < len; ++i) {
        dst[i] = (PGFT_char)(src[i]);
    }

    utf32_buffer->data[len] = 0;
    utf32_buffer->length = len;
    return utf32_buffer;
}

PGFT_String *
_PGFT_EncodePyString(PyObject *obj, int ucs4)
{
    if (PyUnicode_Check(obj))
        return _encode_unicode_string(obj, ucs4);
    else if (PyBytes_Check(obj))
        return _encode_bytes_string(obj);
    else
        PyErr_Format(PyExc_TypeError,
                     "Expected a Unicode or LATIN1 (bytes) string for text:"
                     " got type %.1024s",
                     Py_TYPE(obj)->tp_name);
    return NULL;
}
