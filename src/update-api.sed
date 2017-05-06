# sed Pygame C API update script
#
# Substitution entries have this form:
#
#    s/\(\W\)original\(\W\)/\1replacement\2/g
#
# The bracketing \(\W\) prevents multiple matches for a particular identifier.
# Otherwise, identifiers like 'IntFromObj' might end up as 'pg_pg_IntFromObj'.

# base.c
s/\(\W\)Pg_buffer\(\W\)/\1pg_buffer\2/g
s/\(\W\)PyExc_SDLError\(\W\)/\1pgExc_SDLError\2/g
s/\(\W\)PyGame_RegisterQuit\(\W\)/\1pg_RegisterQuit\2/g
s/\(\W\)IntFromObj\(\W\)/\1pg_IntFromObj\2/g
s/\(\W\)IntFromObjIndex\(\W\)/\1pg_IntFromObjIndex\2/g
s/\(\W\)TwoIntsFromObj\(\W\)/\1pg_TwoIntsFromObj\2/g
s/\(\W\)FloatFromObj\(\W\)/\1pg_FloatFromObj\2/g
s/\(\W\)FloatFromObjIndex\(\W\)/\1pg_FloatFromObjIndex\2/g
s/\(\W\)TwoFloatsFromObj\(\W\)/\1pg_TwoFloatsFromObj\2/g
s/\(\W\)UintFromObj\(\W\)/\1pg_UintFromObj\2/g
s/\(\W\)UintFromObjIndex\(\W\)/\1pg_UintFromObjIndex\2/g
s/\(\W\)PyGame_Video_AutoQuit\(\W\)/\1pgVideo_AutoQuit\2/g
s/\(\W\)PyGame_Video_AutoInit\(\W\)/\1pgVideo_AutoInit\2/g
s/\(\W\)RGBAFromObj\(\W\)/\1pg_RGBAFromObj\2/g
s/\(\W\)PgBuffer_AsArrayInterface\(\W\)/\1pgBuffer_AsArrayInterface\2/g
s/\(\W\)PgBuffer_AsArrayStruct\(\W\)/\1pgBuffer_AsArrayStruct\2/g
s/\(\W\)PgObject_GetBuffer\(\W\)/\1pgObject_GetBuffer\2/g
s/\(\W\)PgBuffer_Release\(\W\)/\1pgBuffer_Release\2/g
s/\(\W\)PgDict_AsBuffer\(\W\)/\1pgDict_AsBuffer\2/g
s/\(\W\)PgExc_BufferError\(\W\)/\1pgExc_BufferError\2/g
s/\(\W\)Py_GetDefaultWindow\(\W\)/\1pg_GetDefaultWindow\2/g
s/\(\W\)Py_SetDefaultWindow\(\W\)/\1pg_SetDefaultWindow\2/g
s/\(\W\)Py_GetDefaultWindowSurface\(\W\)/\1pg_GetDefaultWindowSurface\2/g
s/\(\W\)Py_SetDefaultWindowSurface\(\W\)/\1pg_SetDefaultWindowSurface\2/g

# rect.c
s/\(\W\)PyRectObject\(\W\)/\1pgRectObject\2/g
s/\(\W\)PyRect_AsRect\(\W\)/\1pgRect_AsRect\2/g
s/\(\W\)PyRect_Type\(\W\)/\1pgRect_Type\2/g
s/\(\W\)PyRect_New\(\W\)/\1pgRect_New\2/g
s/\(\W\)PyRect_New4\(\W\)/\1pgRect_New4\2/g
s/\(\W\)GameRect_FromObject\(\W\)/\1pgRect_FromObject\2/g

# rwobject.c
s/\(\W\)RWopsFromObject\(\W\)/\1pgRWopsFromObject\2/g
s/\(\W\)RWopsCheckObject\(\W\)/\1pgRWopsCheckObject\2/g
s/\(\W\)RWopsFromFileObjectThreaded\(\W\)/\1pgRWopsFromFileObjectThreaded\2/g
s/\(\W\)RWopsCheckObjectThreaded\(\W\)/\1pgRWopsCheckObjectThreaded\2/g
s/\(\W\)RWopsEncodeFilePath\(\W\)/\1pgRWopsEncodeFilePath\2/g
s/\(\W\)RWopsEncodeString\(\W\)/\1pgRWopsEncodeString\2/g
s/\(\W\)RWopsFromFileObject\(\W\)/\1pgRWopsFromFileObject\2/g
