#define PHYSICS_COLLISION_INTERNAL

#include "physicsmod.h"
#include "pgphysics.h"

#define CLIP_LB(p,q,u1,u2,ret)         \
    if (IS_NEAR_ZERO(p))               \
        ret = (q < 0) ? 0 : 1;         \
    else                               \
    {                                  \
        double _val = q / p;           \
        if (p < 0)                     \
            u1 = MAX (u1, _val);       \
        else                           \
            u2 = MIN (u2, _val);       \
        ret = 1;                       \
    }

#define MAX_CONTACTS 16
typedef struct
{
    PyVector2 normal;
    PyVector2 contacts[MAX_CONTACTS];
    int       contact_size;
    double    min_depth;
} _Collision;
    
static int _clip_test (AABBox *box, PyVector2 *vertices, int count,
    _Collision *collision);
static void _sat_collision (PyVector2 *pos1, double rot1, PyVector2 *pos2,
    double rot2, AABBox *box1, AABBox *box2, _Collision *collision);

static PyObject* _collide_rect_rect (PyShape* shape1, PyVector2 pos1,
    double rot1, PyShape *shape2, PyVector2 pos2, double rot2);
static PyObject* _collide_rect_circle (PyShape* shape1, PyVector2 pos1,
    double rot1, PyShape *shape2, PyVector2 pos2, double rot2);
static PyObject* _collide_circle_circle (PyShape* shape1, PyVector2 pos1,
    double rot1, PyShape *shape2, PyVector2 pos2, double rot2);

static int
_clip_test (AABBox *box, PyVector2 *vertices, int count, _Collision *collision)
{
    int i, i1, apart = 1, ret;
    PyVector2 dp;
    double u1, u2;
    collision->contact_size = 0;
    
    for (i = 0; i < count; i++)
    {
        i1 = (i + 1) % count;
        u1 = 0.f;
        u2 = 1.f;
        dp = c_diff (vertices[i], vertices[i1]);
        CLIP_LB (-dp.real, vertices[i].real - box->left, u1, u2, ret);
        if (!ret)
            continue;
        CLIP_LB (dp.real, box->right - vertices[i].real, u1, u2, ret);
        if (!ret)
            continue;
        CLIP_LB (-dp.imag, vertices[i].imag - box->bottom, u1, u2, ret);
        if (!ret)
            continue;
        CLIP_LB (dp.imag, box->top - vertices[i].imag, u1, u2, ret);
        if (!ret)
            continue;
        
        if (u1 > u2)
            continue;
        
        apart = 0;
        if (u1 == 0.f)
            collision->contacts[collision->contact_size++] = vertices[i];
        else
            collision->contacts[collision->contact_size++] =
                c_sum (vertices[i], PyVector2_MultiplyWithReal (dp, u1));
        if (u2 == 1.f)
            collision->contacts[collision->contact_size++] = vertices[i1];
        else
            collision->contacts[collision->contact_size++] =
                c_sum (vertices[i], PyVector2_MultiplyWithReal (dp, u2));
    }
    return !apart;
}

static void
_sat_collision (PyVector2 *pos1, double rot1, PyVector2 *pos2, double rot2,
    AABBox *box1, AABBox *box2,  _Collision *collision)
{
/*     int i, k, size, face_id[2]; */
/*     double deps[4], min_dep[2], tmp1, tmp2; */
/*     PyVector2 conts[2][MAX_CONTACTS]; */
/*     AABBox* box[2]; */
/*     PyVector2 refr, incidr; */

/*     /\* */
/*      * Here conts[0][i] represent the contacts calculated in selfBody's local */
/*      * coordinate. */
/*      * conts[1][i] represent the contacts translated to incBody's local */
/*      *  coordinate. */
/*      * then we can rightly get the two minimal depth. */
/*      * */
/*      * The key is whether we appoint which one to be the reference body, the */
/*      * resuting contacts */
/*      * are equivalent only except for different coordinate. but while */
/*      * calculating the penetrating  */
/*      * depth to all the candidate collision face, we must make sure all the */
/*      * contacts are in the */
/*      * same local coordinate at one time. */
/*      *\/ */
/*     for (i = 0; i < collision->contact_size; ++i) */
/*     { */
/*         conts[0][i] = collision->contacts[i]; */
/*         /\* TODO *\/ */
/*         conts[1][i] = PyVector2_Transform (conts[0][i], inc, incr, self, selfr); */
/*     } */
    
/*     box[0] = box1; */
/*     box[1] = box2; */
/*     /\* TODO *\/ */
/*     self[0] = inc[1] = selfBody; */
/*     inc[0] = self[1] = incBody; */

/*     /\* */
/*      * Now we appoint selfBody to be the reference body and incBody */
/*      * to be the incident body for computing min_dep[0]. And vice versa for */
/*      * min_dep[1]. */
/*      *     */
/*      * Since each computation happens in reference body's local coordinate, */
/*      * it's very simple to get the minimal penetrating depth. */
/*      *\/ */
/*     for (k = 0; k <= 1; ++k) */
/*     { */
/*         memset (deps, 0, sizeof(deps)); */
/*         for (i = 0; i < collision->contact_size; ++i) */
/*         { */
/*             deps[CF_LEFT] += fabs (conts[k][i].real - box[k]->left); */
/*             deps[CF_RIGHT] += fabs (box[k]->right - conts[k][i].real); */
/*             deps[CF_BOTTOM] += fabs (conts[k][i].imag - box[k]->bottom); */
/*             deps[CF_TOP] += fabs (box[k]->top - conts[k][i].imag); */
/*         } */
        
/*         min_dep[k] = DBL_MAX; */
/*         for (i = CF_LEFT; i <= CF_TOP; ++i) */
/*         { */
/*             if (min_dep[k] > deps[i]) */
/*             { */
/*                 face_id[k] = i; */
/*                 min_dep[k] = deps[i]; */
/*             } */
/*         } */
/*     } */

/*    /\* */
/*     * If min_dep[0] < min_dep[1], we choose selfBody to be the right reference */
/*     * body and incBody to be the incident one. And vice versa.  */
/*     *\/ */
/*     k = min_dep[0] < min_dep[1] ? 0 : 1; */
    
/*     candi->min_depth = min_dep[k]; */
/*     size = collision->contact_size; */
/*     collision->contact_size = 0; */
    
/*     /\*  */
/*      * Get collision normal according to the collision face */
/*      * and delete the contacts on the collision face. */
/*      *\/ */
/*     switch (face_id[k]) */
/*     { */
/*     case CF_LEFT: */
/*         PyVector2_Set (collision->normal, -1, 0); */
/*         for (i = 0; i < size; ++i) */
/*             if (!PyMath_IsNearEqual(conts[k][i].real, box[k]->left)) */
/*                 collision->contacts[collision->contact_size++] = conts[k][i]; */
/*         break; */
/*     case CF_RIGHT: */
/*         PyVector2_Set(collision->normal, 1, 0); */
/*         for(i = 0; i < size; ++i) */
/*             if(!PyMath_IsNearEqual(conts[k][i].real, box[k]->right)) */
/*                 collision->contacts[collision->contact_size++] = conts[k][i]; */
/*         break; */
/*     case CF_BOTTOM: */
/*         PyVector2_Set(collision->normal, 0, -1); */
/*         for(i = 0; i < size; ++i) */
/*             if(!PyMath_IsNearEqual(conts[k][i].imag, box[k]->bottom)) */
/*                 collision->contacts[collision->contact_size++] = conts[k][i]; */
/*         break; */
/*     case CF_TOP: */
/*         PyVector2_Set(collision->normal, 0, 1); */
/*         for(i = 0; i < size; ++i) */
/*             if(!PyMath_IsNearEqual(conts[k][i].imag, box[k]->top)) */
/*                 collision->contacts[collision->contact_size++] = conts[k][i]; */
/*         break; */
/*     default: */
/*         assert(0); */
/*     } */
/*     /\* */
/*      * We are nearly reaching the destination except for three things: */
/*      * */
/*      * First, collsion normal and contact are in reference body's local */
/*      * coordinate. */
/*      * We must translate them to the global coordinate for easy usage. */
/*      * */
/*      * Second, In the impulse-based collsion reaction formula, we find there */
/*      * is a small */
/*      * part can be precomputed to speed up the total computation. that's so */
/*      * called kFactor. */
/*      * For more information of that you can read Helmut Garstenauer's thesis. */
/*      * */
/*      * Third, we must assign the right referent body and incident body to */
/*      *  ans_ref and ans_inc. */
/*      *\/ */

/*     PyVector2_Rotate(&(collision->normal), self[k]->fRotation); */
/*     for (i = 0; i < collision->contact_size; ++i) */
/*     { */
/*         PyVector2_Rotate(&(collision->contacts[i]), self[k]->fRotation); */
/*         collision->contacts[i] = c_sum(collision->contacts[i], */
/*             self[k]->vecPosition); */
                
/*         /\*precompute KFactor*\/ */
/*         refR = c_diff(collision->contacts[i], self[k]->vecPosition); */
/*         incidR = c_diff(collision->contacts[i], inc[k]->vecPosition); */
/*         tmp1 = PyVector2_Dot(PyVector2_fCross(PyVector2_Cross(refR, */
/*                     collision->normal), refR), collision->normal) */
/*             / ((PyShapeObject*)self[k]->shape)->rInertia; */
/*         tmp2 = PyVector2_Dot(PyVector2_fCross(PyVector2_Cross(incidR, */
/*                     collision->normal), incidR), collision->normal) */
/*             /((PyShapeObject*)inc[k]->shape)->rInertia; */
        
/*         collision->kFactors[i] = 1/self[k]->fMass + 1/inc[k]->fMass + tmp1 + */
/*             tmp2; */
/*     } */
    
/*     *ans_ref = self[k]; */
/*     *ans_inc = inc[k]; */

}

/**
 * The shapes must contains absolute positional values.
 * Returns a list of contacts.
 */
static PyObject*
_collide_rect_rect (PyShape* shape1, PyVector2 pos1, double rot1,
    PyShape *shape2, PyVector2 pos2, double rot2)
{
    AABBox *box1, *box2;
    PyVector2 *vertices1, *vertices2;
    PyVector2 inpos1[4], inpos2[4];
    int count1, count2;
    PyObject *retval;
    _Collision collision;
    PyRectShape *rsh1 = (PyRectShape*) shape1;
    PyRectShape *rsh2 = (PyRectShape*) shape2;

    collision.normal.real = collision.normal.imag = 0;
    
    box1 = PyShape_GetAABBox_FAST (shape1);
    box2 = PyShape_GetAABBox_FAST (shape2);
    if (!box1 || !box2)
    {
        if (box1)
            PyMem_Free (box1);
        if (box2)
            PyMem_Free (box2);
        return NULL;
    }

    if (!AABBox_Overlaps (box1, box2, OVERLAP_ZERO))
    {
        PyMem_Free (box1);
        PyMem_Free (box2);
        Py_RETURN_NONE; /* No collision at all. */
    }

    /* The boxes overlap, do a fine-grained check. */
    vertices1 = PyShape_GetVertices_FAST (shape1, &count1);
    vertices2 = PyShape_GetVertices_FAST (shape2, &count2);
    if (!vertices1 || !vertices2)
    {
        retval = NULL;
        goto back;
    }
    PyVector2_TransformMultiple (vertices2, inpos2, count2, pos1, rot1, pos2,
        rot2);
    PyVector2_TransformMultiple (vertices1, inpos1, count1, pos2, rot2, pos1,
        rot1);
    
    if (!_clip_test (box1, inpos2, count2, &collision))
    {
        retval = Py_None;
        Py_INCREF (retval);
        goto back;
    }

/*     if (AABBox_Contains (&box2, &vertices1[0], 0.f)) */
/*         collision.contacts[collision.contact_size++] = rsh1->bottomleft; */
/*     if (AABBox_Contains (&box2, &vertices1[1], 0.f)) */
/*         collision.contacts[collision.contact_size++] = rsh1->bottomright; */
/*     if (AABBox_Contains (&box2, &vertices1[2], 0.f)) */
/*         collision.contacts[collision.contact_size++] = rsh1->topright; */
/*     if (AABBox_Contains (&box2, &vertices1[3], 0.f)) */
/*         collision.contacts[collision.contact_size++] = rsh1->topleft; */

/*     _sat_collision (&pos1, rot1, &pos2, rot2, &box1, &box2, collision); */

/*     /\* */
/*      * */
/*      *\/ */
/*     pAcc = PyObject_Malloc(sizeof(PyVector2)); */
/*     pAcc->real = pAcc->imag = 0; */
/*     pSplitAcc = PyObject_Malloc(sizeof(PyVector2)); */
/*     pSplitAcc->real = pSplitAcc->imag = 0; */
/*     for(i = 0; i < candi.contact_size; ++i) */
/*     { */
/*         contact = (PyContact*)PyContact_New(ans_ref, ans_inc); */
/*         contact->pos = candi.contacts[i]; */
/*         contact->normal = candi.normal; */

/*         contact->ppAccMoment = PyObject_Malloc(sizeof(PyVector2*)); */
/*         *(contact->ppAccMoment) = pAcc; */
/*         contact->ppSplitAccMoment = PyObject_Malloc(sizeof(PyVector2*)); */
/*         *(contact->ppSplitAccMoment) = pSplitAcc; */

/*         contact->weight = candi.contact_size; */
/*         contact->depth = candi.min_depth; */
/*         contact->kFactor = candi.kFactors[i]; */

/*         PyList_Append(contactList, (PyObject*)contact); */
/*     } */

back:
    if (box1)
        PyMem_Free (box1);
    if (box2)
        PyMem_Free (box2);
    if (vertices1)
        PyMem_Free (vertices1);
    if (vertices2)
        PyMem_Free (vertices2);
    return retval;
}

static PyObject*
_collide_rect_circle (PyShape* shape1, PyVector2 pos1, double rot1,
    PyShape *shape2, PyVector2 pos2, double rot2)
{
    return NULL;
}

static PyObject*
_collide_circle_circle (PyShape* shape1, PyVector2 pos1, double rot1,
    PyShape *shape2, PyVector2 pos2, double rot2)
{
    return NULL;
}

collisionfunc
PyCollision_GetCollisionFunc (ShapeType t1, ShapeType t2, int *swap)
{
    if (!swap)
    {
        PyErr_SetString (PyExc_RuntimeError, "swap argument missing");
        return NULL;
    }

    swap = 0;
    switch (t1)
    {
        case RECT:
            switch (t2)
            {
                case RECT:
                    return _collide_rect_rect;
                case CIRCLE:
                    return _collide_rect_circle;
                default:
                    break;
            }
            break;
        case CIRCLE:
            switch (t2)
            {
                case RECT:
                    *swap = 1;
                    return _collide_rect_circle;
                case CIRCLE:
                    return _collide_circle_circle;
                default:
                    break;
            }
            break;
        default:
            break;
    }
    return NULL;
}
