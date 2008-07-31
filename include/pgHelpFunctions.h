#ifndef _PG_HELPFUNCTIONS_H
#define _PG_HELPFUNCTIONS_H

/**
 * Tries to retrieve the double value from a python object.
 * Taken from the pygame codebase.
 *
 * @param obj The PyObject to get the double from.
 * @param val Pointer to the double to store the value in.
 * @return 0 on failure, 1 on success
 */
int
DoubleFromObj (PyObject* obj, double* val);


PyObject* FromPhysicsVector2ToPygamePoint(pgVector2 v2);

#endif /* _PG_HELPFUNCTIONS_H */
