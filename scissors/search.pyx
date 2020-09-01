# distutils: language = c++
import cython
import numpy as np
from libcpp.vector cimport vector

# keeps information about pixel (x, y)
cdef struct Node:
    int x
    int y
    int active
    int expanded

    int total_cost
    int has_infinite_cost

    Node* next
    Node* prev

cdef struct List:
    int size
    Node* tail

cdef void list_push(Node* node, List* lst):
    if lst[0].tail != NULL:
        lst[0].tail[0].next = node
        node[0].prev = lst[0].tail

    lst[0].tail = node
    node[0].active = True
    lst[0].size+=1

cdef void list_remove_node(Node* node, List* lst):
    if node == lst[0].tail:
        if node[0].prev != NULL:
            node[0].prev[0].next = NULL
            lst[0].tail = node[0].prev
        else:
            lst[0].tail = NULL

    elif node[0].prev == NULL:
        node[0].next[0].prev = NULL
    else:
        node[0].prev[0].next = node[0].next
        node[0].next[0].prev = node[0].prev

    node[0].next = NULL
    node[0].prev = NULL

    node[0].active = False
    lst[0].size-=1

cdef Node * list_pop(List* lst):
    cdef Node* tail = lst[0].tail
    list_remove_node(tail, lst)
    return tail

cdef Node* get_node_ptr(int x, int y, vector[vector[Node]]* storage):
    cdef Node* node = &storage[0][x][y]
    node[0].x = x
    node[0].y = y
    return node

cdef void set_cost(Node* n, long cost):
    n[0].total_cost = cost
    n[0].has_infinite_cost = False

cdef vector[vector[Node]]* make_node_storage(int w, int h):
    return new vector[vector[Node]](w, vector[Node](h, Node(0, 0, False, False, 0, True, NULL, NULL)))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def search(long [:, :, :, :]static_cost, long [:, :, :, :] dynamic_cost,
            int w, int h, int seed_x, int seed_y, int maximum_local_cost):

    # keeps information about all pixels
    cdef vector[vector[Node]]* raw_storage = make_node_storage(w, h)
    cdef Node* seed_point = get_node_ptr(seed_x, seed_y, raw_storage)

    # seed has 0 cost
    set_cost(seed_point, 0)
    # create active list
    cdef vector[List]* active_list = new vector[List](maximum_local_cost, List(0, NULL))
    # put seed point to the first bucket
    cdef int list_index = 0
    list_push(seed_point, &active_list[0][list_index])

    # next node x and next node y, current x, current y
    cdef long [:, :, :] next_node_map = np.zeros((2, w, h), dtype=np.int)

    # tracks the number of active buckets
    cdef int num_of_active_lists = 1

    cdef long tmp_cost = 0
    cdef long last_expanded_cost = 0

    cdef Node* p = NULL
    cdef Node* q = NULL

    cdef int p_x = 0
    cdef int p_y = 0

    cdef int q_x = 0
    cdef int q_y = 0

    # shift indices of neighbors
    cdef int x_shift = 0, y_shift = 0
    # [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    cdef vector[int] y_shifts = [-1, -1, -1, 0, 0, 1, 1, 1]
    cdef vector[int] x_shifts = [-1, 0, 1, -1, 1, -1, 0, 1]

    # while there are unexpanded points
    while num_of_active_lists != 0:
        last_expanded_cost -= 1

        while True:
            last_expanded_cost += 1
            list_index = last_expanded_cost % maximum_local_cost

            if active_list[0][list_index].size != 0:
                break

        p = list_pop(&active_list[0][list_index])
        p_x = p[0].x
        p_y = p[0].y
        #p = get_node_ptr(p_x, p_y, raw_storage)

        # mark 'p' as expanded
        p[0].expanded = True
        last_expanded_cost = p[0].total_cost

        # reduce number of active buckets
        if active_list[0][list_index].size == 0:
            num_of_active_lists -= 1

        # for each neighbour
        for i in range(8):
            x_shift = x_shifts[i]
            y_shift = y_shifts[i]

            if p_y == 0 and y_shift == -1:
                continue
            elif p_y == h - 1 and y_shift == 1:
                continue

            if p_x == 0 and x_shift == -1:
                continue
            elif p_x == w-1 and x_shift == 1:
                continue

            q_x = p_x + x_shift
            q_y = p_y + y_shift
            q = get_node_ptr(q_x, q_y, raw_storage)

            # such that not expanded
            if q[0].expanded:
                continue

            # compute cumulative cost to neighbour
            # TODO fix axes order
            tmp_cost = p[0].total_cost + static_cost[y_shift + 1, x_shift + 1, p_y, p_x]
            tmp_cost += dynamic_cost[y_shift + 1, x_shift + 1, p_y, p_x]

            if q[0].active and (q[0].has_infinite_cost or tmp_cost < q[0].total_cost):
                # remove higher cost neighbor
                list_index = q[0].total_cost % maximum_local_cost
                list_remove_node(q, &active_list[0][list_index])

                 # reduce number of active buckets
                if active_list[0][list_index].size == 0:
                    num_of_active_lists -= 1

            # if neighbour not in list
            if not q.active:
                # assign neighbor’s cumulative cost
                set_cost(q, tmp_cost)
                # place node to the active list
                list_index = q[0].total_cost % maximum_local_cost
                list_push(q, &active_list[0][list_index])

                # set back pointer
                next_node_map[0, q_x, q_y] = p_x
                next_node_map[1, q_x, q_y] = p_y

                # increase number of active buckets
                if active_list[0][list_index].size == 1:
                    num_of_active_lists += 1



    del raw_storage
    del active_list
    return next_node_map
