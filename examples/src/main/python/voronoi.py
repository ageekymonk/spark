from __future__ import print_function
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import math
from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from itertools import combinations
from Queue import PriorityQueue

class KNNJoin1(object):
    """
    Expected format for r_dataset and s_dataset is (vector, data)
    """
    def __init__(self, num_neighbours, r_dataset, s_dataset, num_pivots, sample_percentage):
        self.k = num_neighbours
        self.r_dataset = r_dataset
        self.s_dataset = s_dataset
        self.r_count = self.r_dataset.count()
        self.s_count = self.s_dataset.count()
        self.num_pivots = num_pivots
        self.sample_percentage = sample_percentage

        self.num_pivot_sets = (self.r_count * sample_percentage) / self.num_pivots
        self.r_summary = {}
        self.s_summary = {}
        self.pivots = []
        self.setup()

    def setup(self):
        self.pivots = self.compute_pivots()
        # print(self.pivots)
        pivots = self.pivots

        def find_nearest_pivot(point):
            return min([(idx, math.sqrt(pivot.squared_distance(point))) for idx, pivot in enumerate(pivots)], key=lambda x: x[1])

        def sortPartitionByDistance(iterator):
            return iter(sorted(iterator, key=lambda k_v: k_v[1][2]))

        r_part = self.r_dataset.map(lambda(x): (x, find_nearest_pivot(x[0]))).map(lambda(x): (x[1][0], (x[0][0], x[0][1], x[1][1]))) \
            .partitionBy(self.num_pivots).mapPartitions(sortPartitionByDistance)

        s_part = self.s_dataset.map(lambda(x): (x, find_nearest_pivot(x[0]))).map(lambda(x): (x[1][0], (x[0][0], x[0][1], x[1][1]))) \
            .partitionBy(self.num_pivots).mapPartitions(sortPartitionByDistance)

        self.r_summary, self.s_summary = self.generate_summary_table(r_part, s_part)

        self.ub = self.get_upper_bound(self.r_summary, self.s_summary)
        self.lb = self.get_lower_bound(self.ub, self.r_summary, self.s_summary)

        # print(self.ub)
        # print(self.lb)

        # print(self.compute_knn(r_part, s_part).collect())
        self.compute_knn(r_part, s_part)

    def compute_pivots(self, strategy='random'):

        rdd = self.r_dataset.sample(False, self.sample_percentage, 1).zipWithIndex().map(lambda x: (x[1], x[0]))\
            .partitionBy(self.num_pivot_sets).map(lambda x: x[1])

        if strategy == 'random':
            return self._random_pivot_select(rdd)
        elif strategy == 'furthest':
            self._furthest_pivot_select()
        elif strategy == 'greedy':
            self._greedy_pivot_select()

    def _random_pivot_select(self, rdd):
        """
        RDD is expected to be of the format (vector, data)
        Returns list of pivots of format [(vector, data), ...]
        """
        def _get_total_dist_in_pivot_set(idx, iter):
            total_dist = 0
            for x in combinations(iter, 2):
                v1, v2 = x[0][0], x[1][0]
                if v1 != v2:
                    total_dist = total_dist + math.sqrt(v1.squared_distance(v2))
            return [(idx, total_dist)]
        max_dist_list = rdd.mapPartitionsWithIndex(_get_total_dist_in_pivot_set).collect()
        pivot_list_idx = max_dist_list.index(max(max_dist_list, key=lambda x: x[1]))

        return rdd.mapPartitionsWithIndex(lambda idx, iter: iter if idx == pivot_list_idx else [])\
            .map(lambda x: x[0]).collect()

    def _furthest_pivot_select(self):
        pass

    def _greedy_pivot_select(self):
        pass

    def generate_summary_table(self, r_part, s_part):
        """
        :param r_part: rdd of the form (pid, (vector, data, distance))
        :param s_part: rdd of the form (pid, (vector, data, distance))
        :return: summary dictionary of form {id : {'max_distance': xx, 'min_distance': yy, } , ...}
        """
        r_set_count = r_part.countByKey()
        s_set_count = s_part.countByKey()
        r_summary_dict = {}
        s_summary_dict = {}

        for elem in r_part.aggregateByKey(0, lambda x, y: x if x > y[2] else y[2], lambda x, y: x if x > y else y).collect():
            r_summary_dict[elem[0]] = {'max_distance': elem[1]}
        for elem in r_part.aggregateByKey(0, lambda x, y: x if x < y[2] else y[2], lambda x, y: x if x < y else y).collect():
            r_summary_dict[elem[0]].update({'min_distance': elem[1]})

        for elem in s_part.aggregateByKey(0, lambda x, y: x if x > y[2] else y[2], lambda x, y: x if x > y else y).collect():
            s_summary_dict[elem[0]] = {'max_distance': elem[1]}
        for elem in s_part.aggregateByKey(0, lambda x, y: x if x < y[2] else y[2], lambda x, y: x if x < y else y).collect():
            s_summary_dict[elem[0]].update({'min_distance': elem[1], 'topn': []})

        for k, v in r_set_count.iteritems():
            r_summary_dict[k].update({'count': v})

        def get_top_n(iterator, k=self.k):
            ret_list = []
            try:
                for i in range(k):
                    ret_list.append(next(iterator))
            finally:
                return ret_list

        for part_data in s_part.mapPartitions(get_top_n).glom().collect():
            for idx, elem in enumerate(part_data):
                s_summary_dict[elem[0]]['topn'].append((elem[1][0], elem[1][2]))

        return (r_summary_dict, s_summary_dict)

    def get_upper_bound(self, r_dict, s_dict):
        ub = []
        for i in range(self.num_pivots):
            theta = PriorityQueue()
            for j in range(self.num_pivots):
                for idx, elem in enumerate(s_dict[j]['topn']):
                    dist = r_dict[i]['max_distance'] + math.sqrt(self.pivots[i].squared_distance(self.pivots[j])) + \
                           math.sqrt(elem[0].squared_distance(self.pivots[j]))
                    if theta.qsize() < self.num_pivots:
                        theta.put(-1 * dist)
                    elif theta.queue[0] < (-1 * dist):
                        theta.get()
                        theta.put(-1 * dist)

            ub.append(-1 * theta.get())
        return ub

    def get_lower_bound(self, ub, r_dict, s_dict):
        lb = []
        for i in range(self.num_pivots):
            lb_r = []
            for j in range(self.num_pivots):
                dist = math.sqrt(self.pivots[i].squared_distance(self.pivots[j])) - r_dict[j]['max_distance'] - ub[j]
                if (dist < 0 ):
                    dist = 0
                if dist < r_dict[i]['max_distance'] and dist < s_dict[i]['max_distance']+1:
                    lb_r.append(dist)
                else:
                    lb_r.append(s_dict[i]['max_distance']+1)

            lb.append(lb_r)
            print(lb)
            exit(0)
        return lb

    def compute_knn(self, r_part, s_part):
        def select_knn_for_partition(data, lb=self.lb, num_pivots=self.num_pivots):
            for i in range(num_pivots):
                if lb[data[0]][i] <= data[1][2]:
                    yield (i, data)

        s_part_sel = s_part.flatMap(select_knn_for_partition).repartitionAndSortWithinPartitions(self.num_pivots)

        def compute_single_knn(item, ub=self.ub, pivots=self.pivots, k=self.k):
            num_pivots = len(pivots)
            res_dict = {}
            saved = 0
            for r_elem in item[1][0]:
                theta = ub[item[0]]
                pq = PriorityQueue()
                # print("Size of s_elem is {0}".format(len(list(item[1][1]))))

                for i in range(num_pivots):
                    distToHP = (pivots[i].squared_distance(r_elem[0]) - pivots[item[0]].squared_distance(r_elem[0])) / (
                        2 * math.sqrt(pivots[i].squared_distance(pivots[item[0]])))
                    if distToHP > theta:
                        continue
                    for s_elem in item[1][1]:
                        if s_elem[0] != i:
                            continue
                        dist = math.sqrt(s_elem[1][0].squared_distance(r_elem[0]))
                        # print("R: {0} {1} {2} {3} {4}".format(r_elem[0], theta, s_elem, dist, math.sqrt(pivots[i].squared_distance(r_elem[0]))))
                        if math.sqrt(pivots[i].squared_distance(s_elem[1][0])) > \
                                math.sqrt(pivots[i].squared_distance(r_elem[0])) + theta:
                            saved = saved + 1
                            continue
                        else:
                            dist = math.sqrt(s_elem[1][0].squared_distance(r_elem[0]))
                            if dist <= theta:
                                if pq.qsize() < k:
                                    pq.put((-1 * dist, s_elem[1][0]))
                                else:
                                    pq.put((-1 * dist, s_elem[1][0]))
                                    pq.get()
                                    theta = -1 * pq.queue[0][0]

                res_dict[r_elem] = []
                if pq.qsize() > 0:
                    while not pq.empty():
                        v = pq.get()
                        res_dict[r_elem].append(v)
                        # print("Neighbour: {0} {1}".format(v[0],v[1]))
            print("Saved is {0}".format(saved))
            return res_dict

        print(r_part.cogroup(s_part_sel, self.num_pivots).map(compute_single_knn).collect())

if __name__ == "__main__":
    """
        Usage: voronoi [partitions]
    """
    sc = SparkContext(appName="VoronoiPartition")

    """
    1. Group Random Points into an RDD
    2. Find the best point with max distance
    3. USe the points as Pivot
    """
    if len(sys.argv) > 1:
        iris_data_str = sc.textFile(sys.argv[1])
        iris_data = iris_data_str.map(lambda (x): (Vectors.dense(x.split(',')[0:10]), x.split(',')[-1]))

        # model = KNNJoin(3, iris_data, iris_data, 3)
        KNNJoin1(3, iris_data, iris_data, 200, 1)

        # print(iris_data.mapPartitions(lambda(x): [len(list(x))]).collect())
        # for rdd in iris_data.randomSplit([0.3, 0.3, 0.4]):
        #     print(rdd.mapPartitions(lambda(x): list(x)).collect())
        # print(iris_data.repartition(3).mapPartitions(lambda(x): [list(x)]).collect())
        # print(iris_data.map(lambda(x): (x,x)).partitionBy(3).mapPartitions(lambda(x): [list(x)]).collect())
    else:
        print("Input the data file")
    sc.stop()
