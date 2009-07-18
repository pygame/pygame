#define PHYSICS_COLLISION_INTERNAL

#include "physicsmod.h"
#include "pgphysics.h"

#define CLIP_LB(p,q,u1,u2,ret)         \
    if (IS_NEAR_ZERO(p))               \
        ret = (q < 0) ? 0 : 1;         \
    else                               \
    {                                  \
        if (p < 0)                     \
            u1 = MAX (u1, q / p);      \
        else                           \
            u2 = MIN (u2, q / p);      \
        ret = 1;                       \
    }

typedef enum
{
    CF_LEFT,
    CF_BOTTOM,
    CF_RIGHT,
    CF_TOP
} CollisionFace;

#define MAX_CONTACTS 16
typedef struct
{
    PyVector2 normal;
    PyVector2 contacts[MAX_CONTACTS];
    int       contact_size;
    int       refno;
    double    min_depth;
} _Collision;
    
static int _clip_test (AABBox *box, PyVector2 *vertices, int count,
    _Collision *collision);
static void _sat_collision (PyVector2 *pos1, double rot1, PyVector2 *pos2,
    double rot2, AABBox *box1, AABBox *box2, _Collision *collision);

static PyObject* _collide_rect_rect (PyShape* shape1, PyVector2 pos1,
    double rot1, PyShape *shape2, PyVector2 pos2, double rot2, int *refid);
static PyObject* _collide_rect_circle (PyShape* shape1, PyVector2 pos1,
    double rot1, PyShape *shape2, PyVector2 pos2, double rot2, int *refid);
static PyObject* _collide_circle_circle (PyShape* shape1, PyVector2 pos1,
    double rot1, PyShape *shape2, PyVector2 pos2, double rot2, int *refid);

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
        
        CLIP_LB (dp.real, vertices[i].real - box->left, u1, u2, ret);
        if (!ret)
            continue;
        CLIP_LB (-dp.real, box->right - vertices[i].real, u1, u2, ret);
        if (!ret)
            continue;
        CLIP_LB (dp.imag, vertices[i].imag - box->bottom, u1, u2, ret);
        if (!ret)
            continue;
        CLIP_LB (-dp.imag, box->top - vertices[i].imag, u1, u2, ret);
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
    AABBox *box1, AABBox *box2, _Collision *collision)
{
    int i, k, size, face_id[2]; 
    double deps[4], min_dep[2];
    PyVector2 conts[2][MAX_CONTACTS];
    AABBox* box[2];
    PyVector2 ppos[2], incpos[2];
    double prot[2], incrot[2];
    
    /**
     * pos1 = body1 pos
     * pos2 = body2 pos
     */
     
    /*
     * Here conts[0][i] represent the contacts calculated in selfBody's local
     * coordinate.
     * conts[1][i] represent the contacts translated to incBody's local
     *  coordinate.
     * then we can rightly get the two minimal depth.
     *
     * The key is whether we appoint which one to be the reference body, the
     * resuting contacts
     * are equivalent only except for different coordinate. but while
     * calculating the penetrating
     * depth to all the candidate collision face, we must make sure all the
     * contacts are in the
     * same local coordinate at one time.
     */
    for (i = 0; i < collision->contact_size; ++i)
    {
        conts[0][i] = collision->contacts[i];
        /* TODO: Maybe invert pos1/pos2 */
        conts[1][i] = PyVector2_Transform (conts[0][i], *pos1, rot1,
            *pos2, rot2);
    }
    
    box[0] = box1;
    box[1] = box2;
    ppos[0] = incpos[1] = *pos1;
    prot[0] = incrot[1] = rot1;
    ppos[1] = incpos[0] = *pos2;
    prot[1] = incrot[0] = rot2;

    /*
     * Now we appoint selfBody to be the reference body and incBody
     * to be the incident body for computing min_dep[0]. And vice versa for
     * min_dep[1].
     *
     * Since each computation happens in reference body's local coordinate,
     * it's very simple to get the minimal penetrating depth.
     */
    for (k = 0; k <= 1; ++k)
    {
        memset (deps, 0, sizeof (deps));
        for (i = 0; i < collision->contact_size; ++i)
        {
            deps[CF_LEFT] += fabs (conts[k][i].real - box[k]->left);
            deps[CF_RIGHT] += fabs (box[k]->right - conts[k][i].real);
            deps[CF_BOTTOM] += fabs (conts[k][i].imag - box[k]->bottom);
            deps[CF_TOP] += fabs (box[k]->top - conts[k][i].imag);
        }
        
        min_dep[k] = DBL_MAX;
        for (i = CF_LEFT; i <= CF_TOP; ++i)
        {
            if (min_dep[k] > deps[i])
            {
                face_id[k] = i;
                min_dep[k] = deps[i];
            }
        }
    }
    
    /*
     * If min_dep[0] < min_dep[1], we choose selfBody to be the right reference
     * body and incBody to be the incident one. And vice versa.
     */
    collision->refno = k = min_dep[0] < min_dep[1] ? 0 : 1;
    collision->min_depth = min_dep[k];
    size = collision->contact_size;
    collision->contact_size = 0;
    
    /*
     * Get collision normal according to the collision face
     * and delete the contacts on the collision face.
     */
    switch (face_id[k])
    {
    case CF_LEFT:
        PyVector2_Set (collision->normal, -1, 0);
        for (i = 0; i < size; ++i)
            if (!PyMath_IsNearEqual(conts[k][i].real, box[k]->left))
                collision->contacts[collision->contact_size++] = conts[k][i];
        break;
    case CF_RIGHT:
        PyVector2_Set (collision->normal, 1, 0);
        for(i = 0; i < size; ++i)
            if(!PyMath_IsNearEqual(conts[k][i].real, box[k]->right))
                collision->contacts[collision->contact_size++] = conts[k][i];
        break;
    case CF_BOTTOM:
        PyVector2_Set (collision->normal, 0, -1);
        for (i = 0; i < size; ++i)
            if (!PyMath_IsNearEqual(conts[k][i].imag, box[k]->bottom))
                collision->contacts[collision->contact_size++] = conts[k][i];
        break;
    case CF_TOP:
        PyVector2_Set (collision->normal, 0, 1);
        for (i = 0; i < size; ++i)
            if (!PyMath_IsNearEqual(conts[k][i].imag, box[k]->top))
                collision->contacts[collision->contact_size++] = conts[k][i];
         break;
    default:
        assert (0);
        break;
     }

    /*
     * We are nearly reaching the destination except for three things:
     *
     * First, collsion normal and contact are in reference body's local
     * coordinate.
     * We must translate them to the global coordinate for easy usage.
     *
     * Second, In the impulse-based collsion reaction formula, we find there
     * is a small
     * part can be precomputed to speed up the total computation. that's so
     * called kFactor.
     * For more information of that you can read Helmut Garstenauer's thesis.
     */
    PyVector2_Rotate (&(collision->normal), prot[k]);
    for (i = 0; i < collision->contact_size; ++i)
    {
        PyVector2_Rotate (&(collision->contacts[i]), prot[k]);
        collision->contacts[i] = c_sum (collision->contacts[i], ppos[k]);
    }
}

/**
 * The shapes must contains absolute positional values.
 * Returns a list of contacts.
 */
static PyObject*
_collide_rect_rect (PyShape* shape1, PyVector2 pos1, double rot1,
    PyShape *shape2, PyVector2 pos2, double rot2, int *refid)
{
    AABBox box1, box2;
    PyVector2 *vertices1, *vertices2;
    PyVector2 inpos1[4], inpos2[4];
    int count1, count2, i;
    PyObject *retval;
    PyContact *contact;
    _Collision collision;
    PyRectShape *rsh1 = (PyRectShape*) shape1;
    PyRectShape *rsh2 = (PyRectShape*) shape2;

    if (!shape1 || !shape2)
        return NULL;

    collision.normal.real = collision.normal.imag = 0;
    
    if (!PyShape_GetAABBox_FAST (shape1, &box1) ||
        !PyShape_GetAABBox_FAST (shape2, &box2))
        return NULL;

    if (!AABBox_Overlaps (&box1, &box2, OVERLAP_ZERO))
    {
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

    /* TODO: problem here! */
    PyVector2_TransformMultiple (vertices1, inpos1, count1, pos2, rot2, pos1,
        rot1);
    PyVector2_TransformMultiple (vertices2, inpos2, count2, pos1, rot1, pos2,
        rot2);

/*
    for (i = 0; i < count1; i++)
    {
        printf ("vertices1, %d: %.3f, %.3f\n", i, inpos1[i].real,
            inpos1[i].imag);
    }
    puts ("---");
    for (i = 0; i < count2; i++)
    {
        printf ("vertices2, %d: %.3f, %.3f\n", i, vertices2[i].real,
            vertices2[i].imag);
    }
    puts ("---");
*/

    if (!_clip_test (&box1, inpos2, count2, &collision))
    {
        retval = Py_None;
        Py_INCREF (retval);
        goto back;
    }

    puts ("DSA");
    if (AABBox_Contains (&box2, &vertices1[0], 0.f))
        collision.contacts[collision.contact_size++] = rsh1->bottomleft;
    if (AABBox_Contains (&box2, &vertices1[1], 0.f))
        collision.contacts[collision.contact_size++] = rsh1->bottomright;
    if (AABBox_Contains (&box2, &vertices1[2], 0.f))
        collision.contacts[collision.contact_size++] = rsh1->topright;
    if (AABBox_Contains (&box2, &vertices1[3], 0.f))
        collision.contacts[collision.contact_size++] = rsh1->topleft;

    _sat_collision (&pos1, rot1, &pos2, rot2, &box1, &box2, &collision);

    /*
     *
     */
    retval = PyList_New (0);
    if (!retval)
        goto back;

    for (i = 0; i < collision.contact_size; ++i)
    {
        contact = (PyContact*) PyContact_New ();
        if (!contact)
        {
            Py_DECREF (retval);
            retval = NULL;
            goto back;
        }
        
        contact->position = collision.contacts[i];
        contact->normal = collision.normal;
        PyVector2_Set (contact->acc_moment, 0, 0);
        PyVector2_Set (contact->split_acc_moment, 0, 0);
        contact->weight = collision.contact_size;
        contact->depth = collision.min_depth;
        
        if (PyList_Append (retval, (PyObject*)contact) == -1)
        {
            Py_DECREF (retval);
            retval = NULL;
            goto back;
        }
        Py_DECREF ((PyObject*)contact);
    }
    
    *refid = collision.refno;
    
back:
    if (vertices1)
        PyMem_Free (vertices1);
    if (vertices2)
        PyMem_Free (vertices2);
    return retval;
}

static PyObject*
_collide_rect_circle (PyShape* shape1, PyVector2 pos1, double rot1,
    PyShape *shape2, PyVector2 pos2, double rot2, int *refid)
{
    return NULL;
}

static PyObject*
_collide_circle_circle (PyShape* shape1, PyVector2 pos1, double rot1,
    PyShape *shape2, PyVector2 pos2, double rot2, int *refid)
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
        case ST_RECT:
            switch (t2)
            {
                case ST_RECT:
                    return _collide_rect_rect;
                case ST_CIRCLE:
                    return _collide_rect_circle;
                default:
                    break;
            }
            break;
        case ST_CIRCLE:
            switch (t2)
            {
                case ST_RECT:
                    *swap = 1;
                    return _collide_rect_circle;
                case ST_CIRCLE:
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
