/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.examples

import breeze.linalg.squaredDistance
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner
import org.apache.log4j.Logger
import org.apache.spark
import org.apache.spark.SparkException
import org.apache.spark.ml.{PredictorParams, PredictionModel, Predictor}
import org.apache.spark.ml.param.{ParamMap, ParamValidators, Param, DoubleParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.{PartitionPruningRDD, RDD}
import org.apache.spark.sql.DataFrame
import scala.collection.mutable
import org.apache.spark.HashPartitioner



/**
 * Naive Bayes Classifiers.
 * It supports both Multinomial NB
 * ([[http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html]])
 * which can handle finitely supported discrete data. For example, by converting documents into
 * TF-IDF vectors, it can be used for document classification. By making every vector a
 * binary (0/1) data, it can also be used as Bernoulli NB
 * ([[http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html]]).
 * The input feature values must be nonnegative.
 */
class KNearestNeighbour(override val uid: String)
  extends Predictor[Vector, KNearestNeighbour, KNearestNeighbourModel] {

  var metric: String = "euclidean"

  def this() = this(Identifiable.randomUID("knn"))

  /**
   * Set the metric type using a string (case-sensitive).
   * Supported options: "euclidean" and "manhattan".
   * Default is "euclidean"
   */
  def setMetric(value: String): this.type = {
    this.metric = value
    this
  }

  override protected def train(dataset: DataFrame): KNearestNeighbourModel = {
    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset)
    new KNearestNeighbourModel(uid, metric, null)
  }

  def train(dataset: RDD[LabeledPoint]): KNearestNeighbourModel = {
    new KNearestNeighbourModel(uid, metric, dataset)
  }

  override def toString: String = {
    s"KNearestNeighbour classes"
  }

  override def copy(extra: ParamMap): KNearestNeighbour = defaultCopy(extra)
}

/**
 * Model produced by [[KNearestNeighbour]]
 */
class KNearestNeighbourModel(override val uid: String,
                              val distMetric: String,
                              val trainData: RDD[LabeledPoint],
                              val numNeighbours: Int = 3,
                              val numPivots: Int = 3
                              )
  extends PredictionModel[Vector, KNearestNeighbourModel] {

  var pivots:Array[Vector] = null
  var featureSummary:List[mutable.Map[String, Any]] = null
  var trainSummary:List[mutable.Map[String, Any]] = null
  var trainDataLen = 0L
  var featureDataLen = 0L

  override def predict(features: Vector): Double = {
    0.0
  }

  def findNearestPivotIndex(pivots:Array[Vector], fvector: Vector): Int = {
    pivots.zipWithIndex.map(
      (v) => (v._2, Vectors.sqdist(v._1, fvector))).minBy((v) => v._2)._1
  }

  def generateSummaryTable(pFeatures: RDD[(Int, (Vector, Double))], pTrain: RDD[(Int, (LabeledPoint, Double))]) = {

    featureSummary = List.fill(pivots.length)(mutable.Map[String, Any]())
    trainSummary = List.fill(pivots.length)(mutable.Map[String, Any]())

    pFeatures.glom().map( v => (v.maxBy(_._2._2), v.minBy(_._2._2), v.length)).collect().foreach( (elem) =>
      featureSummary(elem._1._1) += ("max_distance" -> elem._1._2._2, "min_distance" -> elem._2._2._2, "count" -> elem._3)
    )

    pTrain.glom().map( v => (v.maxBy(_._2._2), v.minBy(_._2._2), v.take(numNeighbours))).collect().foreach( (elem) =>
      trainSummary(elem._1._1) += ("max_distance" -> elem._1._2._2, "min_distance" -> elem._2._2._2, "topn" -> elem._3.toList)
    )

    println(featureSummary)
  }

  def getUpperBound() = {

  }

  def getLowerBound() = {

  }

  def predict(features: RDD[Vector]): Array[Vector] = {

    trainDataLen = trainData.count()
    featureDataLen = features.count()
    pivots = computePivots(features)

    println("Number of pivots calculated = " + pivots.length.toString)

    // pfeatures is the voronoi partitioned RDD. Number of partition = number of pivots
    // Each partition is sorted by Distance from pivot
    // Each element is tuple of (pivot_index, (vector, distance_to_pivot))
    val pfeatures = features.map( (fvector) =>
      (pivots.zipWithIndex.map((v) =>
        (v._2, (fvector, Vectors.sqdist(v._1, fvector)))).minBy((v) => v._2._2))
    ).partitionBy(new spark.HashPartitioner(pivots.length)).mapPartitions( (iter) => iter.toList.sortBy( f => f._2._2).toIterator)

    // pTrain is the voronoi partitioned RDD. Number of partition = number of pivots
    // Each partition is sorted by Distance from pivot
    // Each element is tuple of (pivot_index, (labeled_point, distance_to_pivot))
    val pTrain = trainData.map( (lpoint) => (pivots.zipWithIndex.map((v) =>
      (v._2, (lpoint, Vectors.sqdist(v._1, lpoint.features)))).minBy((v) => v._2._2))
    ).partitionBy(new spark.HashPartitioner(pivots.length)).mapPartitions( (iter) => iter.toList.sortBy( f => f._2._2).toIterator)

    pTrain.cache()

    generateSummaryTable(pfeatures, pTrain)

    println("ERROR: Number of partitions in pfeatures = " + pfeatures.partitions.size.toString)
    println("ERROR: Number of partitions in pTrain = " + pTrain.partitions.size.toString)

    // Join  => (pivot index ((vector, distance to pivot), (labeled point, distance to pivot) )  )
    // map  => (vector, (labeled point, pivot index))
    // groupByKey => (vector, [(labeled point, pivot index)])
    // map => (vector, (idx, [N Neighbour labeled point, distance ])

//    val pFeatureWithLN = pfeatures.join(pTrain).map( (v) =>
//      (v._2._1._1, (v._2._2._1, Vectors.sqdist(v._2._1._1, v._2._2._1.features), v._1))
//    ).groupByKey().flatMap( v => List((v._1, v._2.toArray.sortBy(_._2).take(numNeighbours).toList)))

    // pFeatureWithLN is a tuple of (feature_vector, Array(Labeled_Point, distance_to_feature_vec, partition_id))
    val pFeatureWithLN = pTrain.glom().mapPartitionsWithIndex( (idx, iter) => {
      val elemList = iter.next()
      List((elemList(0)._1, elemList)).toIterator
    }
    ).join(pfeatures).map( (v) => {
      val elemList = v._2._1
      val rElem = v._2._2
      (rElem._1, elemList.map((elem) => (elem._2._1, Vectors.sqdist(elem._2._1.features, rElem._1), elem._1)).sortBy(_._2).take(numNeighbours).toList)
    }
    )
    println("ERROR: Number of partitions in pFeatureWithLN = " + pFeatureWithLN.partitions.size.toString)

    // (featureVector, Neighbours)
    // pFWithLNAndRN is a tuple of (feature_vector, Array(Labeled_Point, distance_to_feature_vec, partition_id), List(partition_id_to_check)
    val pFWithLNAndRN = pFeatureWithLN.map( (v) =>
      (v._1, v._2, pivots.zipWithIndex.flatMap({
        case (p, idx) =>
          if ((p != v._1) && (p != pivots(v._2(0)._3))) {
            val dist = (Vectors.sqdist(p, v._1) - Vectors.sqdist(v._1, pivots(v._2(0)._3))) / (2 * math.sqrt(Vectors.sqdist(p, pivots(v._2(0)._3))))
            if ((dist >= math.sqrt(v._2.last._2)) ||
              (math.sqrt(Vectors.sqdist(p, v._1)) - trainSummary(idx)("max_distance").asInstanceOf[Double] > dist)) {
              None
            }
            else {
              List((math.sqrt(v._2.last._2) - dist, idx))
            }
          }
          else {
            None
          }
      }
      ).sortBy(_._1).reverse.map(_._2).toList, v._2(0)._3
    ))

    println("ERROR: KNN Found within same partition = " + pFWithLNAndRN.filter((v) => v._3.length == 0).count())

    var pf = pFWithLNAndRN.filter(v => v._3.length > 0)

    while(pf.count() > 0)
    {
      val pFilteredR = pf
      // Move R data to another partition
      val pFilteredRWithPart = pFilteredR.map(v => (v._3.head, (v._1, v._2, v._3.tail, v._4)))

      // Find N Neighbours greater than last
//      val newRWithLN = pFilteredRWithPart.join(pTrain).flatMap( (v) => {
//        val dist = Vectors.sqdist(v._2._1._1, v._2._2._1.features)
//        if (dist > v._2._1._2.last._2) {
//          List((v._2._1._1,(v._2._1._2.last, v._2._1._2, v._2._1._3, v._2._1._4)))
//        }
//        else
//        {
//          // vector, nn, nnlist, nextpart
//          List((v._2._1._1,((v._2._2._1, dist, v._1), v._2._1._2, v._2._1._3, v._2._1._4)))
//        }
//      }).groupByKey().flatMap( v => {
//        val nn = v._2.head._2 ::: v._2.flatMap( (pts) => List(pts._1)).toList
//        nn.sortBy(_._2).take(numNeighbours).toList
//        List((v._1, nn, v._2.head._3, v._2.head._4))
//      })

      val newRWithLN = pTrain.glom().mapPartitionsWithIndex( (idx, iter) => {
        val elemList = iter.next()
        List((elemList(0)._1, elemList)).toIterator
      }
      ).join(pFilteredRWithPart).map( (v) => {
        val elemList = v._2._1
        val (rElem, rElemNbrs, rElemPart, idx) = v._2._2
        (rElem, rElemNbrs :::  elemList.flatMap((elem) => {
          val dist = Vectors.sqdist(elem._2._1.features, rElem)
          if (dist < rElemNbrs.last._2)
            List((elem._2._1, dist, elem._1))
          else
            None
        }
        ).sortBy(_._2).take(numNeighbours).toList, rElemPart, idx)
      })


      // (featureVector, Neighbours)
      val newRWithLNAndRN = newRWithLN.map( (v) =>
        (v._1, v._2, v._3.flatMap({
          case (idx) =>
            val p = pivots(idx)
            if (p != v._1) {
              val dist = (Vectors.sqdist(p, v._1) - Vectors.sqdist(v._1, pivots(v._4))) / (2 * math.sqrt(Vectors.sqdist(p, pivots(v._4))))
              if ((dist >= math.sqrt(v._2.last._2)) ||
                (math.sqrt(Vectors.sqdist(p, v._1)) - trainSummary(idx)("max_distance").asInstanceOf[Double] > dist)) {
                None
              }
              else {
                List((math.sqrt(v._2.last._2) - dist, idx))
              }
            }
            else {
              None
            }
        }).sortBy(_._1).reverse.map(_._2).toList, v._4)
      )
      println("ERROR: KNN Found within same partition = " + newRWithLNAndRN.filter((v) => v._3.length == 0).count())

      pf = newRWithLNAndRN.filter(v => v._3.length > 0)
    }

    pivots
  }

  // Select pivots from a random sample groups based on max distance between each other
  protected def randomPivotSelect(rData: RDD[Vector]) = {

    val pivot_distance = rData.mapPartitionsWithIndex { (idx, iter) =>
      val total_dist = iter.toSeq.combinations(2).foldLeft(0.0) { (dist, vecs) =>
        dist + Vectors.sqdist(vecs(0), vecs(1))
      }
      List((idx, total_dist)).toIterator
    }

    val partition_idx = pivot_distance.max()(Ordering.by(t => t._2))._1
    new PartitionPruningRDD[Vector](rData, (idx) => idx == partition_idx).collect()
  }

  protected def kmeansPivotSelect(RData: RDD[Vector]) = {

  }

  def computePivots(RData: RDD[Vector], strategy: String = "random"): Array[Vector] = {

    println("ERROR: Starting to compute pivots")
    val pRData = RData.sample(false, 1.0, 101).zipWithIndex().map( v => (v._2, v._1))
      .partitionBy(new spark.HashPartitioner((featureDataLen/numPivots).asInstanceOf[Int])).map(_._2)
    println("ERROR: Data is Partitioned")
    strategy match  {
      case "random" => randomPivotSelect(pRData)
//      case "greedy" => kmeansPivotSelect(pRData)
    }

  }

  override def copy(extra: ParamMap): KNearestNeighbourModel = {
    copyValues(new KNearestNeighbourModel(uid, distMetric, trainData).setParent(this.parent), extra)
  }

  override def toString: String = {
    s"KNearestNeighbour classes"
  }

}
