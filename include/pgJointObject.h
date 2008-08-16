/*
  pygame physics - Pygame physics module

  Copyright (C) 2008 Zhang Fan

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

/**
* Joint solve algorithm
* Velocity after free updating may violate joint constraint,
* Solve joint velocity to set velocity right,
* body position may violate joint constraint by error accumulation and discrete time step simulation,
* so body position must be corrected.
* *********************
* Distance joint solve method
* joint anchor velocity is vp, body velocity is vb, body angular velocity is wb,the vector between body anchor
* to body center is l
* vp = vb + cross(wb,l)
* d(vp) = d(vb) + cross(d(wb),l)
* I is the impulse which joint gives to body. I = m*d(vb), cross(I,l) = Interia*(wb)
* the constraint is distance(anchor1-anchor2) = constant,it can apply this:
* project((vp1 - vp2),(anchor1-anchor2)) = 0
* so,we can solve it by the above equations.
*/

#ifndef _PHYSICS_JOINT_H_
#define _PHYSICS_JOINT_H_

/**
 * Solve joint velocity constraint.
 *
 * @param joint
 * @param stepTime
 */
int JointObject_SolveConstraintVelocity (PyJointObject *joint, double stepTime);

/**
 * Solve joint position constraint.
 *
 * @param joint
 * @param stepTime
 */
int JointObject_SolveConstraintPosition (PyJointObject *joint, double stepTime);

/**
 * Python C API export hook.
 *
 * @param c_api Pointer to the C API array.
 */
void PyJointObject_ExportCAPI (void **c_api);

#endif /* _PHYSICS_JOINT_H_ */
