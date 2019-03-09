/* Auto generated file: with makeref.py .  Docs go in docs/reST/ref/ . */
#define DOC_PYGAMERECT "Rect(left, top, width, height) -> Rect\nRect((left, top), (width, height)) -> Rect\nRect(object) -> Rect\npygame object for storing rectangular coordinates"
#define DOC_RECTCOPY "copy() -> Rect\ncopy the rectangle"
#define DOC_RECTMOVE "move(x, y) -> Rect\nmoves the rectangle"
#define DOC_RECTMOVEIP "move_ip(x, y) -> None\nmoves the rectangle, in place"
#define DOC_RECTINFLATE "inflate(x, y) -> Rect\ngrow or shrink the rectangle size"
#define DOC_RECTINFLATEIP "inflate_ip(x, y) -> None\ngrow or shrink the rectangle size, in place"
#define DOC_RECTCLAMP "clamp(Rect) -> Rect\nmoves the rectangle inside another"
#define DOC_RECTCLAMPIP "clamp_ip(Rect) -> None\nmoves the rectangle inside another, in place"
#define DOC_RECTCLIP "clip(Rect) -> Rect\ncrops a rectangle inside another"
#define DOC_RECTUNION "union(Rect) -> Rect\njoins two rectangles into one"
#define DOC_RECTUNIONIP "union_ip(Rect) -> None\njoins two rectangles into one, in place"
#define DOC_RECTUNIONALL "unionall(Rect_sequence) -> Rect\nthe union of many rectangles"
#define DOC_RECTUNIONALLIP "unionall_ip(Rect_sequence) -> None\nthe union of many rectangles, in place"
#define DOC_RECTFIT "fit(Rect) -> Rect\nresize and move a rectangle with aspect ratio"
#define DOC_RECTNORMALIZE "normalize() -> None\ncorrect negative sizes"
#define DOC_RECTCONTAINS "contains(Rect) -> bool\ntest if one rectangle is inside another"
#define DOC_RECTCOLLIDEPOINT "collidepoint(x, y) -> bool\ncollidepoint((x,y)) -> bool\ntest if a point is inside a rectangle"
#define DOC_RECTCOLLIDERECT "colliderect(Rect) -> bool\ntest if two rectangles overlap"
#define DOC_RECTCOLLIDELIST "collidelist(list) -> index\ntest if one rectangle in a list intersects"
#define DOC_RECTCOLLIDELISTALL "collidelistall(list) -> indices\ntest if all rectangles in a list intersect"
#define DOC_RECTCOLLIDEDICT "collidedict(dict) -> (key, value)\ntest if one rectangle in a dictionary intersects"
#define DOC_RECTCOLLIDEDICTALL "collidedictall(dict) -> [(key, value), ...]\ntest if all rectangles in a dictionary intersect"


/* Docs in a comment... slightly easier to read. */

/*

pygame.Rect
 Rect(left, top, width, height) -> Rect
 Rect((left, top), (width, height)) -> Rect
 Rect(object) -> Rect
pygame object for storing rectangular coordinates

pygame.Rect.copy
 copy() -> Rect
copy the rectangle

pygame.Rect.move
 move(x, y) -> Rect
moves the rectangle

pygame.Rect.move_ip
 move_ip(x, y) -> None
moves the rectangle, in place

pygame.Rect.inflate
 inflate(x, y) -> Rect
grow or shrink the rectangle size

pygame.Rect.inflate_ip
 inflate_ip(x, y) -> None
grow or shrink the rectangle size, in place

pygame.Rect.clamp
 clamp(Rect) -> Rect
moves the rectangle inside another

pygame.Rect.clamp_ip
 clamp_ip(Rect) -> None
moves the rectangle inside another, in place

pygame.Rect.clip
 clip(Rect) -> Rect
crops a rectangle inside another

pygame.Rect.union
 union(Rect) -> Rect
joins two rectangles into one

pygame.Rect.union_ip
 union_ip(Rect) -> None
joins two rectangles into one, in place

pygame.Rect.unionall
 unionall(Rect_sequence) -> Rect
the union of many rectangles

pygame.Rect.unionall_ip
 unionall_ip(Rect_sequence) -> None
the union of many rectangles, in place

pygame.Rect.fit
 fit(Rect) -> Rect
resize and move a rectangle with aspect ratio

pygame.Rect.normalize
 normalize() -> None
correct negative sizes

pygame.Rect.contains
 contains(Rect) -> bool
test if one rectangle is inside another

pygame.Rect.collidepoint
 collidepoint(x, y) -> bool
 collidepoint((x,y)) -> bool
test if a point is inside a rectangle

pygame.Rect.colliderect
 colliderect(Rect) -> bool
test if two rectangles overlap

pygame.Rect.collidelist
 collidelist(list) -> index
test if one rectangle in a list intersects

pygame.Rect.collidelistall
 collidelistall(list) -> indices
test if all rectangles in a list intersect

pygame.Rect.collidedict
 collidedict(dict) -> (key, value)
test if one rectangle in a dictionary intersects

pygame.Rect.collidedictall
 collidedictall(dict) -> [(key, value), ...]
test if all rectangles in a dictionary intersect

*/