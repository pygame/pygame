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

# color.c
s/\(\W\)PyColor_Type\(\W\)/\1pgColor_Type\2/g
s/\(\W\)PyColor_New\(\W\)/\1pgColor_New\2/g
s/\(\W\)RGBAFromColorObj\(\W\)/\1pg_RGBAFromColorObj\2/g
s/\(\W\)PyColor_NewLength\(\W\)/\1pgColor_NewLength\2/g

# bufferproxy.c
s/\(\W\)PgBufproxy_Type\(\W\)/\1pgBufproxy_Type\2/g
s/\(\W\)PgBufproxy_New\(\W\)/\1pgBufproxy_New\2/g
s/\(\W\)PgBufproxy_GetParent\(\W\)/\1pgBufproxy_GetParent\2/g
s/\(\W\)PgBufproxy_Trip\(\W\)/\1pgBufproxy_Trip\2/g
s/\(\W\)PgBufproxy_Check\(\W\)/\1pgBufproxy_Check\2/g

# surflock.c
s/\(\W\)PyLifetimeLock\(\W\)/\1pgLifetimeLockObject\2/g
s/\(\W\)PyLifetimeLock_Check\(\W\)/\1pgLifetimeLock_Check\2/g
s/\(\W\)PySurface_Prep\(\W\)/\1pgSurface_Prep\2/g
s/\(\W\)PySurface_Unprep\(\W\)/\1pgSurface_Unprep\2/g
s/\(\W\)PySurface_Lock\(\W\)/\1pgSurface_Lock\2/g
s/\(\W\)PySurface_Unlock\(\W\)/\1pgSurface_Unlock\2/g
s/\(\W\)PySurface_LockBy\(\W\)/\1pgSurface_LockBy\2/g
s/\(\W\)PySurface_UnlockBy\(\W\)/\1pgSurface_UnlockBy\2/g
s/\(\W\)PySurface_LockLifetime\(\W\)/\1pgSurface_LockLifetime\2/g

# surface.c
s/\(\W\)PySurfaceObject\(\W\)/\1pgSurfaceObject\2/g
s/\(\W\)PySurface_Type\(\W\)/\1pgSurface_Type\2/g
s/\(\W\)PySurface_New\(\W\)/\1pgSurface_New\2/g
s/\(\W\)PySurface_NewNoOwn\(\W\)/\1pgSurface_NewNoOwn\2/g
s/\(\W\)PySurface_Check\(\W\)/\1pgSurface_Check\2/g
s/\(\W\)PySurface_AsSurface\(\W\)/\1pgSurface_AsSurface\2/g
s/\(\W\)PySurface_Blit\(\W\)/\1pgSurface_Blit\2/g

# event.c
s/\(\W\)PyEventObject\(\W\)/\1pgEventObject\2/g
s/\(\W\)PyEvent_Check\(\W\)/\1pgEvent_Check\2/g
s/\(\W\)PyEvent_Type\(\W\)/\1pgEvent_Type\2/g
s/\(\W\)PyEvent_New\(\W\)/\1pgEvent_New\2/g
s/\(\W\)PyEvent_New2\(\W\)/\1pgEvent_New2\2/g
s/\(\W\)PyEvent_FillUserEvent\(\W\)/\1pgEvent_FillUserEvent\2/g
s/\(\W\)Py_EnableKeyRepeat\(\W\)/\1pg_EnableKeyRepeat\2/g
s/\(\W\)Py_GetKeyRepeat\(\W\)/\1pg_GetKeyRepeat\2/g

# display.c
s/\(\W\)PyVidInfoObject\(\W\)/\1pgVidInfoObject\2/g
s/\(\W\)PyVidInfo_Type\(\W\)/\1pgVidInfo_Type\2/g
s/\(\W\)PyVidInfo_New\(\W\)/\1pgVidInfo_New\2/g
s/\(\W\)PyVidInfo_AsVidInfo\(\W\)/\1pgVidInfo_AsVidInfo\2/g
s/\(\W\)PyVidInfo_Check\(\W\)/\1pgVidInfo_Check\2/g

# mixer.c
s/\(\W\)PySoundObject\(\W\)/\1pgSoundObject\2/g
s/\(\W\)PyChannelObject\(\W\)/\1pgChannelObject\2/g
s/\(\W\)PySound_AsChunk\(\W\)/\1pgSound_AsChunk\2/g
s/\(\W\)PyChannel_AsInt\(\W\)/\1pgChannel_AsInt\2/g
s/\(\W\)PySound_Check\(\W\)/\1pgSound_Check\2/g
s/\(\W\)PySound_Type\(\W\)/\1pgSound_Type\2/g
s/\(\W\)PySound_New\(\W\)/\1pgSound_New\2/g
s/\(\W\)PySound_Play\(\W\)/\1pgSound_Play\2/g
s/\(\W\)PyChannel_Check\(\W\)/\1pgChannel_Check\2/g
s/\(\W\)PyChannel_Type\(\W\)/\1pgChannel_Type\2/g
s/\(\W\)PyChannel_New\(\W\)/\1pgChannel_New\2/g
s/\(\W\)PyMixer_AutoInit\(\W\)/\1pgMixer_AutoInit\2/g
s/\(\W\)PyMixer_AutoQuit\(\W\)/\1pgMixer_AutoQuit\2/g
