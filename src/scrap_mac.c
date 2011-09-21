/*
    pygame - Python Game Library
    Copyright (C) 2006, 2007 Rene Dudfield, Marcus von Appen

    Originally written and put in the public domain by Sam Lantinga.

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

static PyObject*
mac_scrap_call (char *name, PyObject *args)
{
    static PyObject *mac_scrap_module = NULL;
    PyObject *method;
    PyObject *result;

    if (!mac_scrap_module)
        mac_scrap_module = PyImport_ImportModule ("pygame.mac_scrap");
    if (!mac_scrap_module)
        return NULL;
    
    method = PyObject_GetAttrString (mac_scrap_module, name);
    if (!method)
        return NULL;
    result = PyObject_CallObject (method, args);
    Py_DECREF (method);
    return result;
}

static PyObject*
_scrap_init (PyObject *self, PyObject *args)
{
    return mac_scrap_call ("init", args);
}

static PyObject*
_scrap_get_scrap (PyObject *self, PyObject *args)
{
    return mac_scrap_call ("get", args);
}

static PyObject*
_scrap_put_scrap (PyObject *self, PyObject *args)
{
    return mac_scrap_call ("put", args);
}

static PyObject*
_scrap_lost_scrap (PyObject *self, PyObject *args)
{
    return mac_scrap_call ("lost", args);
}

static PyObject*
_scrap_get_types (PyObject *self, PyObject *args)
{
    return mac_scrap_call ("get_types", args);
}

static PyObject*
_scrap_contains (PyObject *self, PyObject *args)
{
    return mac_scrap_call ("contains", args);
}

static PyObject*
_scrap_set_mode (PyObject *self, PyObject *args)
{
    return mac_scrap_call ("set_mode", args);
}
