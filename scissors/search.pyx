# distutils: language = c++
import cython
import numpy as np
from libcpp.vector cimport vector

from cpython cimport array
import array

# keeps information about pixel (x, y)
cdef struct Node:
    int active
    int expanded

    int total_cost
    int has_infinite_cost


cdef Node* get_node_ptr(int x, int y, vector[vector[Node]]* storage):
    return &storage[0][x][y]

cdef void set_cost(Node* n, long cost):
    n[0].total_cost = cost
    n[0].has_infinite_cost = False

cdef void toggle_activation(Node* n):
    n[0].active = ~n[0].active

cdef vector[vector[Node]]* make_node_storage(int w, int h):
    return new vector[vector[Node]](w, vector[Node](h, Node(False, False, 0, True)))



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def search(long [:, :, :, :]local_cost, int w, int h, int seed_x, int seed_y, int maximum_local_cost):
    # keeps information about all pixels
    cdef vector[vector[Node]]* raw_storage = make_node_storage(w, h)
    cdef Node* seed_point = get_node_ptr(seed_x, seed_y, raw_storage)

    # seed has 0 cost
    set_cost(seed_point, 0)
    # create active list
    cdef list active_list = [[] for _ in range(maximum_local_cost)]
    # put seed point to the first bucket
    cdef int list_index = 0
    active_list[list_index].append((seed_x, seed_y))

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
    cdef list shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # while there are unexpanded points
    while num_of_active_lists != 0:
        last_expanded_cost -= 1

        while True:
            last_expanded_cost += 1
            list_index = last_expanded_cost % maximum_local_cost

            if len(active_list[list_index]) != 0:
                break

        p_x, p_y = active_list[list_index].pop()
        p = get_node_ptr(p_x, p_y, raw_storage)

        # mark 'p' as expanded
        p[0].expanded = True
        last_expanded_cost = p[0].total_cost

        # reduce number of active buckets
        if len(active_list[list_index]) == 0:
            num_of_active_lists -= 1

        # for each neighbour
        for y_shift, x_shift in shifts:
            if p_y == 0 or p_x == 0 or p_x == w - 1 or p_y == h - 1:
                continue

            q_x = p_x + x_shift
            q_y = p_y + y_shift
            q = get_node_ptr(q_x, q_y, raw_storage)

            # such that not expanded
            if q[0].expanded:
                continue

            # compute cumulative cost to neighbour
            tmp_cost = p[0].total_cost + local_cost[y_shift + 1, x_shift + 1, p_y, p_x]

            if q[0].active and (q[0].has_infinite_cost or tmp_cost < q[0].total_cost):
                # remove higher cost neighbor
                list_index = q[0].total_cost % maximum_local_cost
                active_list[list_index].remove((q_x, q_y))
                toggle_activation(q)

                 # reduce number of active buckets
                if len(active_list[list_index]) == 0:
                    num_of_active_lists -= 1

            # if neighbour not in list
            if not q.active:
                # assign neighbor’s cumulative cost
                set_cost(q, tmp_cost)
                # place node to the active list
                list_index = q[0].total_cost % maximum_local_cost
                active_list[list_index].append((q_x, q_y))
                toggle_activation(q)

                # set back pointer
                next_node_map[0, q_x, q_y] = p_x
                next_node_map[1, q_x, q_y] = p_y

                # increase number of active buckets
                if len(active_list[list_index]) == 1:
                    num_of_active_lists += 1

    del raw_storage
    return next_node_map
