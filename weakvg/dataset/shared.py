def get_locations(queries):
    """
    Return a list locations for each query, where a location is a list of
    6 elements [left, right, center, top, bottom, middle]
    """
    n_locations = 6

    left = ["left", "leftmost", "leftmost part", "left part", "far left"]
    right = ["right", "rightmost", "rightmost part", "right part", "far right"]
    center = ["center", "middle", "middle part", "center part"]
    top = ["top", "topmost", "topmost part", "top part", "above", "over"]
    bottom = [
        "bottom",
        "bottommost",
        "bottommost part",
        "bottom part",
        "below",
        "under",
    ]
    middle = ["middle", "middle part", "center part"]

    has_location = lambda query, keywords: int(
        any([keyword in query for keyword in keywords])
    )

    get_location = lambda query: (
        [
            has_location(query, left),
            has_location(query, right),
            has_location(query, center),
            has_location(query, top),
            has_location(query, bottom),
            has_location(query, middle),
        ]
    )

    locations = [get_location(query) for query in queries]

    # avoid filtering out queries with no location
    locations = [
        location if sum(location) > 0 else [1 for _ in range(n_locations)]
        for location in locations
    ]

    return locations


def get_relations(proposals, labels):
    """
    Return a list of relations for each proposal, where a relation is a list of
    6 elements [left, right, center, top, bottom, middle] indicating whether the
    box has one relations wrt the other boxes of the same class
    """
    n_relations = 6

    relations = [[1 for _ in range(n_relations)] for _ in range(len(proposals))]

    get_center = lambda x: ((x[0] + x[2]) / 2, (x[1] + x[3]) / 2)

    indexes = [i for i in range(len(proposals))]
    centers = [get_center(box) for box in proposals]

    for label in set(labels):
        indexes_by_label = [i for i in indexes if labels[i] == label]
        centers_by_label = [centers[i] for i in indexes_by_label]

        leftmost = min(centers_by_label, key=lambda x: x[0])[0]  # x
        rightmost = max(centers_by_label, key=lambda x: x[0])[0]  # x
        topmost = min(centers_by_label, key=lambda x: x[1])[1]  # y
        bottommost = max(centers_by_label, key=lambda x: x[1])[1]  # y

        if len(indexes_by_label) > 1:
            for box_index in indexes_by_label:
                cx, cy = centers[box_index]

                relations[box_index][0] = 1 if cx == leftmost else 0
                relations[box_index][1] = 1 if cx == rightmost else 0
                relations[box_index][2] = 1 if cx != leftmost and cx != rightmost else 0
                relations[box_index][3] = 1 if cy == topmost else 0
                relations[box_index][4] = 1 if cy == bottommost else 0
                relations[box_index][5] = 1 if cy != topmost and cy != bottommost else 0

    return relations
