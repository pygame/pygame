#include "pgAABBBox.h"

pgAABBBox PG_GenAABB(double left, double right, double bottom, double top)
{
	pgAABBBox box;
	box.left = left;
	box.right = right;
	box.bottom = bottom;
	box.top = top;
	return box;
}
