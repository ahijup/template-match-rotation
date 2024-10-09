from typing import List, Tuple

Point = Tuple[float, float]
Polygon = List[Point]

def is_inside(p: Point, edge_start: Point, edge_end: Point) -> bool:
    # Determine if point p is inside the half-plane defined by the edge (edge_start, edge_end)
    x0, y0 = p
    x1, y1 = edge_start
    x2, y2 = edge_end
    # Compute the cross product
    cross = (x2 - x1)*(y0 - y1) - (y2 - y1)*(x0 - x1)
    return cross >= 0  # Corrected the inequality

def compute_intersection(s: Point, e: Point, cp1: Point, cp2: Point) -> Point:
    # Compute the intersection point of the lines (s, e) and (cp1, cp2)
    x1, y1 = s
    x2, y2 = e
    x3, y3 = cp1
    x4, y4 = cp2

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        # Lines are parallel (should not happen in convex polygons if edges are not colinear)
        return None

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return (px, py)

def sutherland_hodgman(subject_polygon: Polygon, clip_polygon: Polygon) -> Polygon:
    output_list = subject_polygon
    cp1 = clip_polygon[-1]

    for cp2 in clip_polygon:
        input_list = output_list
        output_list = []
        if not input_list:
            # All points have been clipped
            break
        s = input_list[-1]

        for e in input_list:
            if is_inside(e, cp1, cp2):
                if not is_inside(s, cp1, cp2):
                    # Compute and add intersection point
                    inter_pt = compute_intersection(s, e, cp1, cp2)
                    if inter_pt:
                        output_list.append(inter_pt)
                output_list.append(e)
            elif is_inside(s, cp1, cp2):
                # Compute and add intersection point
                inter_pt = compute_intersection(s, e, cp1, cp2)
                if inter_pt:
                    output_list.append(inter_pt)
            s = e
        cp1 = cp2

    return output_list

def polygon_area(polygon: Polygon) -> float:
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0
