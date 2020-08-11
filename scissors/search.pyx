# distutils: language = c++

from libcpp.vector cimport vector

import cython

cdef struct Node:
    int active
    int expanded

    int x
    int y
    Node* next_node
    long total_cost


cdef class NodeStorage:
    cdef vector[vector[Node]]* storage
    cdef vector[vector[int]]* was_used

    def __cinit__(self, int w, int h):
        self.storage = new vector[vector[Node]](w, vector[Node](h))
        self.was_used = new vector[vector[int]](w, vector[int](h, False))

    def __dealloc__(self):
        del self.storage
        del self.was_used

    cdef Node* get_node(self, int x, int y):
        if not self.was_used[0][x][y]:
            self.storage[0][x][y].active = False
            self.storage[0][x][y].expanded = False
            self.storage[0][x][y].next_node = NULL
            self.storage[0][x][y].total_cost = 100000000
            self.storage[0][x][y].x = x
            self.storage[0][x][y].y = y

            self.was_used[0][x][y] = True

        return &self.storage[0][x][y]

@cython.boundscheck(False)
@cython.wraparound(False)
def search(long [:, :, :, :]local_cost, int w, int h, int seed_x, int seed_y, int free_x, int free_y, int maximum_local_cost):

    storage = NodeStorage(w, h)
    cdef Node* seed_point = storage.get_node(seed_x, seed_y)

    seed_point[0].active = True
    seed_point[0].total_cost = 0

    cdef list active_list = [[] for _ in range(maximum_local_cost)]
    active_list[0].append((seed_point[0].x, seed_point[0].y))

    cdef long last_expanded_cost = 0
    cdef long tmp_cost = 0
    cdef Node* p = NULL
    cdef Node* q = NULL
    cdef int tmp_counter

    cdef int tmp_x
    cdef int tmp_y

    cdef int x_shift, y_shift
    cdef list shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while any(len(l) != 0 for l in active_list):
        last_expanded_cost -= 1

        while True:
            last_expanded_cost += 1
            tmp_counter = last_expanded_cost % maximum_local_cost

            if len(active_list[tmp_counter]) != 0:
                break

        tmp_x, tmp_y = active_list[tmp_counter].pop()
        p = storage.get_node(tmp_x, tmp_y)

        p[0].expanded = True
        last_expanded_cost = p[0].total_cost

        for y_shift, x_shift in shifts:
            if p[0].y == 0 or p[0].x == 0 or p[0].x == w - 1 or p[0].y == h - 1:
                continue

            q = storage.get_node(p[0].x + x_shift, p[0].y + y_shift)

            if q[0].expanded:
                continue

            tmp_cost = p[0].total_cost + local_cost[y_shift + 1, x_shift + 1, p[0].y, p[0].x]

            if q[0].active and tmp_cost < q[0].total_cost:
                tmp_counter = q[0].total_cost % maximum_local_cost

                active_list[tmp_counter].remove((q[0].x, q[0].y))
                q[0].active = False

            if not q.active:
                q[0].total_cost = tmp_cost
                q[0].next_node = p
                tmp_counter = q[0].total_cost % maximum_local_cost

                active_list[tmp_counter].append((q[0].x, q[0].y))
                q[0].active = True

    history = []
    cur_point = storage.get_node(free_x, free_y)
    while cur_point[0].next_node != NULL:
        history.append((cur_point[0].y, cur_point[0].x))
        cur_point = cur_point[0].next_node

    return history
