# sed Pygame C API update script
#
# When adding new entries, watch out for identifiers like
# IntFromObjIndex and IntFromObj. If they are added as separate
# sed commands, then for the longer identifier there will be
# two matches and the prefix gets added twice (eg pg_pg_).
# Either use just the shorter pattern or combine like this:
#
#   s/IntFromObj\(\|Index\)/pg_IntFromObj\1/g

# base.c
s/Pg_buffer/pg_buffer/g
s/PyExc_SDLError/pgExc_SDLError/g
s/PyGame_RegisterQuit/pg_RegisterQuit/g
s/IntFromObj\(\|Index\)/pg_IntFromObj\1/g
s/TwoIntsFromObj/pg_TwoIntsFromObj/g
s/FloatFromObj\(\|Index\)/pg_FloatFromObj\1/g
s/TwoFloatsFromObj/pg_TwoFloatsFromObj/g
s/UintFromObj\(\|Index\)/pg_UintFromObj\1/g
s/PyGame_Video_AutoQuit/pgVideo_AutoQuit/g
s/PyGame_Video_AutoInit/pgVideo_AutoInit/g
s/RGBAFromObj/pg_RGBAFromObj/g
s/PgBuffer_AsArrayInterface/pgBuffer_AsArrayInterface/g
s/PgBuffer_AsArrayStruct/pgBuffer_AsArrayStruct/g
s/PgObject_GetBuffer/pgObject_GetBuffer/g
s/PgBuffer_Release/pgBuffer_Release/g
s/PgDict_AsBuffer/pgDict_AsBuffer/g
s/PgExc_BufferError/pgExc_BufferError/g
s/Py_GetDefaultWindow/pg_GetDefaultWindow/g
s/Py_SetDefaultWindow/pg_SetDefaultWindow/g
s/Py_GetDefaultWindowSurface/pg_GetDefaultWindowSurface/g
s/Py_SetDefaultWindowSurface/pg_SetDefaultWindowSurface/g
