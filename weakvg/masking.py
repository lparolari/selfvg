def get_queries_mask(queries):
    """
    Return a mask for the words and queries in a batch of queries.

    :param queries: A tensor with shape `[b, q, w]`
    :return: A tuple of tensors `([b, q, w], [b, q])` for is_word, is_query
    """
    is_word = queries != 0  # [b, q, w]
    is_query = is_word.any(-1)  # [b, q]

    return is_word, is_query


def get_queries_mask_(x):
    """
    Return a mask for the words and queries in a batch of queries.

    :param x: A dict with keys `queries`
    :return: A tuple of tensors `([b, q, w], [b, q])` for is_word, is_query
    """
    queries = x["queries"]

    return get_queries_mask(queries)


def get_queries_count(queries):
    """
    Return the number of words and queries in a batch of queries.

    :param queries: A tensor with shape `[b, q, w]`
    :return: A tuple of tensors `([b, q], [b])` for n_words, n_queries
    """
    is_word, is_query = get_queries_mask(queries)

    n_words = is_word.sum(-1)  # [b, q]
    n_queries = is_query.sum(-1)  # [b]

    return n_words, n_queries


def get_proposals_mask(proposals):
    """
    Return a mask for the proposals in a batch of proposals.

    :param proposals: A tensor with shape `[b, p, 4]`
    :return: A tensor with shape `[b, p]`
    """
    return proposals.greater(0).any(-1)


def get_proposals_mask_(x):
    """
    Return a mask for the proposals in a batch of proposals.

    :param x: A dict with keys `proposals`
    :return: A tensor with shape `[b, p]`
    """
    proposals = x["proposals"]

    return get_proposals_mask(proposals)


def get_multimodal_mask(queries, proposals):
    """
    :param queries: A tensor with shape `[b, q, w]`
    :param proposals: A tensor with shape `[b, p, 4]`
    :return: A tensor with shape `[b, q, b, p]`
    """
    queries_mask = get_queries_mask(queries)[1]  # [b, q]
    proposals_mask = get_proposals_mask(proposals)  # [b, p]

    b = queries.shape[0]
    q = queries.shape[1]
    p = proposals.shape[1]

    queries_mask = queries_mask.view(b, q, 1, 1).repeat(1, 1, b, p)  # [b, q, b, p]
    proposals_mask = proposals_mask.view(1, 1, b, p).repeat(b, q, 1, 1)  # [b, q, b, p]

    return queries_mask & proposals_mask


def get_concepts_mask(heads, labels):
    """
    :param heads: A tensor with shape `[b, q, h]`
    :param labels: A tensor with shape `[b, p]`
    :return: A tensor with shape `[b, q, b, p]`
    """
    n_heads = (heads != 0).sum(-1).unsqueeze(-1)  # [b, q, 1]

    has_head = (n_heads != 0).unsqueeze(-1)  # [b, q, 1, 1]

    has_label = (labels != 0).unsqueeze(0).unsqueeze(0)  # [1, 1, b, p]

    mask = has_head & has_label

    return mask


def get_concepts_mask_(x):
    """
    :param x: A dict with keys `heads`, `labels`
    :return: A tensor with shape `[b, q, b, p]`
    """
    heads = x["heads"]
    labels = x["labels"]

    return get_concepts_mask(heads, labels)


def get_mask(queries, proposals, heads, labels):
    """
    :param x: A dict with keys `queries`, `proposals`, `heads`, `labels`
    :return: A tensor with shape `[b, q, b, p]`
    """
    multimodal_mask = get_multimodal_mask(queries, proposals)
    concepts_mask = get_concepts_mask(heads, labels)

    return multimodal_mask & concepts_mask


def get_mask_(x):
    """
    :param x: A dict with keys `queries`, `proposals`, `heads`, `labels`
    :return: A tensor with shape `[b, q, b, p]`
    """
    queries = x["queries"]
    proposals = x["proposals"]
    heads = x["heads"]
    labels = x["labels"]

    return get_mask(queries, proposals, heads, labels)


def get_relations_mask(locations, relations):
    """
    :param locations: A tensor with shape `[b, q, r]`
    :param relations: A tensor with shape `[b, p, r]`
    :return: A boolean tensor with shape `[b, q, b, p]`
    """
    b = locations.shape[0]
    q = locations.shape[1]
    p = relations.shape[1]

    locations = locations.reshape(b, q, 1, 1, -1).repeat(1, 1, b, p, 1)
    relations = relations.reshape(1, 1, b, p, -1).repeat(b, q, 1, 1, 1)

    spatial = locations * relations  # [b, q, b, p, r]
    spatial = spatial.sum(-1)  # [b, q, b, p]
    spatial = spatial >= 1  # [b, q, b, p]

    return spatial


def get_relations_mask_(x):
    """
    :param x: A dict with keys `locations`, `relations`
    :return: A boolean tensor with shape `[b, q, b, p]`
    """
    locations = x["locations"]
    relations = x["relations"]

    return get_relations_mask(locations, relations)
