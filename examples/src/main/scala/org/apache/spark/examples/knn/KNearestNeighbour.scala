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

import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.{PartitionPruningRDD, RDD}
import org.apache.spark.sql.DataFrame
import scala.collection.immutable.TreeMap
import scala.collection.mutable
import org.apache.spark.Partitioner
import org.apache.spark.SparkContext
import scala.util.Random

class ExactPartitioner[V](partitions: Int)
  extends Partitioner {

  def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[Int]
    return k % partitions
  }

  override def numPartitions: Int = {
    return partitions
  }
}

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
                              var numPivots: Int = 3
                              )
  extends PredictionModel[Vector, KNearestNeighbourModel] {

  var featureSummary:List[mutable.Map[String, Any]] = null
  var trainSummary:List[mutable.Map[String, Any]] = null

  var trainDataLen = trainData.count()
  println("ERROR: Total train data = " + trainDataLen.toString)
  trainData.cache()
  var pivots:Array[Vector] = computePivots(trainData)
  numPivots = pivots.length
  var delta = 50
  var featureDataLen = 0L

   /*
   pTrain is the voronoi partitioned RDD. Number of partition = number of pivots
   Each partition is sorted by Distance from pivot
   Each element is tuple of (pivot_index, (labeled_point, distance_to_pivot))
   */
  val pTrain = trainData.map( (lpoint) =>
     pivots.zipWithIndex.map((v) =>
      (v._2, (lpoint, math.sqrt(Vectors.sqdist(v._1, lpoint.features))))).minBy((v) => v._2._2)
   )
   .partitionBy(new ExactPartitioner(numPivots))
   .mapPartitions( (iter) => iter.toList.sortBy( f => f._2._2 ).toIterator )

  pTrain.cache()

  generateSummaryTable(pTrain)

  val pTrainList = pTrain.glom().filter(f => f.length > 0).mapPartitionsWithIndex( (idx, iter) => {
    val elemList = iter.next()
    List((elemList(0)._1, elemList)).toIterator
  }
  )

  pTrainList.cache()
  pTrain.unpersist()


//  val mapTrain = pTrain.glom().map((v) => {
//    var treemap = TreeMap.empty[Double, List[LabeledPoint]]
//
//    v.foreach(elem => {
//      if (treemap.contains(math.sqrt(elem._2._2))) {
//        treemap += (math.sqrt(elem._2._2) -> (treemap(math.sqrt(elem._2._2)) ::: List(elem._2._1)))
//      }
//      else
//      {
//        treemap += (math.sqrt(elem._2._2) -> List(elem._2._1))
//      }
//
//    })
//    (v(0)._1, treemap)
//  })

  override def predict(features: Vector): Double = {
    0.0
  }

  def generateSummaryTable(pTrain: RDD[(Int, (LabeledPoint, Double))]) = {

    trainSummary = List.fill(pivots.length)(mutable.Map[String, Any]())

    pTrain.glom().flatMap( v => {
      if (v.isEmpty)
      {
        None
      }
      else
      Some((v.maxBy(_._2._2), v.minBy(_._2._2)))
    }
    )
      .collect().foreach( (elem) =>
      trainSummary(elem._1._1) += ("max_distance" -> elem._1._2._2, "min_distance" -> elem._2._2._2)
    )

  }

  def predict(rData: RDD[Vector], pathName: String) = {

    featureDataLen = rData.count()

     /*
     pfeatures is the voronoi partitioned RDD. Number of partition = number of pivots
     Within each partition elements are sorted by Distance from pivot
     Each element is tuple of (pivot_index, (vector, distance_to_pivot))
     */
    val pfeatures = rData.map( (fVector) =>
      pivots.zipWithIndex.map({
        case(pVector, idx) => (idx, (fVector, math.sqrt(Vectors.sqdist(pVector, fVector))))
      }).minBy((v) => v._2._2)
    )
     .partitionBy(new ExactPartitioner(numPivots))
     .mapPartitions( (iter) => iter.toList.toIterator)
//       .mapPartitions( (iter) => iter.toList.sortBy( f => f._2._2).toIterator)

    // Join  => (pivot index ((vector, distance to pivot), (labeled point, distance to pivot) )  )
    // map  => (vector, (labeled point, pivot index))
    // groupByKey => (vector, [(labeled point, pivot index)])
    // map => (vector, (idx, [N Neighbour labeled point, distance ])

    // pFeatureWithLN is a tuple of (feature_vector, Array(Labeled_Point, distance_to_feature_vec, partition_id), feature_vector_partition_id)
    val pFeatureWithLN = pTrainList.join(pfeatures).map( (v) => {
      val elemList = v._2._1
      val rElem = v._2._2
      (rElem._1, elemList.map((elem) => (elem._2._1, math.sqrt(Vectors.sqdist(elem._2._1.features, rElem._1)), elem._1)).sortBy(_._2).take(numNeighbours).toList, v._1)
    })

//    val xyz =  mapTrain.join(pfeatures).map(v => {
//      val elemMap = v._2._1
//      val rElem = v._2._2
//      val distFromPivot = math.sqrt(Vectors.sqdist(rElem._1, pivots(v._1)))
//      var computedNeighbours = List.empty[(LabeledPoint, Double, Int)]
//      var numNeighboursFound = 0
//      var distStartLeft = math.floor(distFromPivot)
//      var distStartRight = math.floor(distFromPivot)
//      var mindist = delta
//
//      while((numNeighboursFound < numNeighbours) && ((distStartRight <= math.sqrt(trainSummary(v._1).getOrElse("max_distance", 0).asInstanceOf[Double])) || (distStartLeft >= 0.0))) {
//
//        if (distStartLeft >= 0.0){
//          val leftData = elemMap.range(distStartLeft-delta, distStartLeft)
//          computedNeighbours = computedNeighbours ::: leftData.flatMap(elem =>  elem._2.map(p => (p, Vectors.sqdist(p.features, rElem._1), v._1))).toList
//        }
//
//        val rightData = elemMap.range(distStartRight, distStartRight+delta)
//        computedNeighbours = computedNeighbours ::: rightData.flatMap(elem =>  elem._2.map(p => (p, Vectors.sqdist(p.features, rElem._1), v._1))).toList
//
//        numNeighboursFound = computedNeighbours.count(elem => elem._2 <= mindist)
//
//        distStartLeft = distStartLeft - delta
//        distStartRight = distStartRight + delta
//        mindist = mindist + delta
//      }
//
//      (rElem._1, computedNeighbours.sortBy(_._2).distinct.take(numNeighbours), v._1)
//
//    })

    // pFWithLNAndRN is a tuple of (feature_vector, Array(Labeled_Point, distance_to_feature_vec, partition_id), List(partition_id_to_check), feature_vector_partition_id)
    val pFWithLNAndRN = pFeatureWithLN.map( (v) => {
      val part_list = pivots.zipWithIndex.flatMap({
        case (p, idx) =>
          if ((p != v._1) && (v._2.length > 0) && (p != pivots(v._2(0)._3))) {
            val dist = (Vectors.sqdist(p, v._1) - Vectors.sqdist(v._1, pivots(v._2(0)._3))) / (2 * math.sqrt(Vectors.sqdist(p, pivots(v._2(0)._3))))
            if ((dist >= v._2.last._2) ||
              ((! trainSummary(idx).isEmpty) && (math.sqrt(Vectors.sqdist(p, v._1)) - trainSummary(idx)("max_distance").asInstanceOf[Double] > v._2.last._2))) {
              List()
            }
            else {
              List((v._2.last._2 - dist, idx))
            }
          }
          else {
            List()
          }
      })
      if (part_list.length == 0) {
        (v._1, v._2, List(), v._3)
      }
      else
      {
        (v._1, v._2, part_list.sortBy(_._1).reverse.map(_._2).toList, v._3)
//        (v._1, v._2, part_list.sortBy(_._2).reverse.map(_._2).toList, v._3)
      }
    })

    pFWithLNAndRN.cache()
    pFWithLNAndRN.count()

    var result = pFWithLNAndRN.filter((v)=> v._3.length == 0).map(v => (v._1, v._2.flatMap(e => Some((e._1, e._2)))))
    result.cache()

    try
    {
      println("ERROR: Number of completed elements = " + result.count())
    }
    catch
    {
      case msg:Throwable => println("ERROR: Number of completed elements = 0")
    }

    var pf = pFWithLNAndRN.filter(v => v._3.length > 0)

    pf.cache()
    pf.count()

    var total_repl = pf.map((v) => v._3.size).reduce((tot, add) => tot + add)

    pFWithLNAndRN.unpersist()

    while(total_repl > trainDataLen)
    {
      val pFilteredR = pf
      // Move R data to another partition
      val pFilteredRWithPart = pFilteredR.map(v => (v._3.head, (v._1, v._2, v._3.tail, v._4))).partitionBy(new ExactPartitioner(numPivots))

      // newRWithLN is a tuple of (feature_vector, Array(Labeled_Point, distance_to_feature_vec, partition_id), feature_vector_partition_id)
      val newRWithLN = pTrainList.join(pFilteredRWithPart).map( (v) => {
        val elemList = v._2._1
        val (rElem, rElemNbrs, rElemPart, idx) = v._2._2
        val rElemNbrsUpdated = rElemNbrs :::  elemList.flatMap((elem) => {
          val dist = math.sqrt(Vectors.sqdist(elem._2._1.features, rElem))
          if (dist < rElemNbrs.last._2)
            List((elem._2._1, dist, elem._1))
          else
            None
        }).toList
        (rElem, rElemNbrsUpdated.sortBy(_._2).take(numNeighbours), rElemPart, idx)
      })

      // newRWithLNAndRN is a tuple of (feature_vector, Array(Labeled_Point, distance_to_feature_vec, partition_id), List(partition_id_to_check), feature_vector_partition_id)
      val newRWithLNAndRN = newRWithLN.map( (v) =>
        (v._1, v._2, v._3.flatMap({
          case (idx) =>
            val p = pivots(idx)
            if (p != v._1) {
              val dist = (Vectors.sqdist(p, v._1) - Vectors.sqdist(v._1, pivots(v._4))) / (2 * math.sqrt(Vectors.sqdist(p, pivots(v._4))))
              if ((dist >= v._2.last._2) ||
                ((! trainSummary(idx).isEmpty) && (math.sqrt(Vectors.sqdist(p, v._1)) - trainSummary(idx)("max_distance").asInstanceOf[Double] > v._2.last._2))) {
                None
              }
              else {
                List((v._2.last._2 - dist, idx))
              }
            }
            else {
              None
            }
        }).sortBy(_._1).reverse.map(_._2), v._4)
      )

      newRWithLNAndRN.cache()
      newRWithLNAndRN.count()
      pf.unpersist()

      var prev_result = result

      result = result.union(newRWithLNAndRN.filter((v)=> v._3.length == 0).map(v => (v._1, v._2.flatMap(e => Some((e._1, e._2))))))
      result.cache()
      println("ERROR: Total number of computed elements  =" + result.count())

      prev_result.unpersist()

      pf = newRWithLNAndRN.filter(v => v._3.length > 0)
      pf.cache()
      println("ERROR: Total number of Remaining elements  =" + pf.count())
      newRWithLNAndRN.unpersist()

      total_repl = pf.map((v) => v._3.size).reduce((tot, add) => tot + add)
      println("ERROR: Total elems when replicated = " + total_repl.toString )

    }

    val pFilteredR = pf
    // Replicated part
    val pReplicatedRWithPart = pFilteredR.flatMap(v =>
      v._3.map( (elem) => (elem, (v._1, v._2, None, v._4)))
    ).partitionBy(new ExactPartitioner(numPivots))

    // newRWithLN is a tuple of (feature_vector, Array(Labeled_Point, distance_to_feature_vec, partition_id), feature_vector_partition_id)
    val newRWithLN = pTrainList.join(pReplicatedRWithPart).map( (v) => {
      val elemList = v._2._1
      val (rElem, rElemNbrs, rElemPart, idx) = v._2._2
      val rElemNbrsUpdated = rElemNbrs :::  elemList.flatMap((elem) => {
        val dist = math.sqrt(Vectors.sqdist(elem._2._1.features, rElem))
        if (dist < rElemNbrs.last._2)
          List((elem._2._1, dist, elem._1))
        else
          None
      }).toList
      (rElem, rElemNbrsUpdated.sortBy(_._2).take(numNeighbours), rElemPart, idx)
    })

    // Combine all the results together
    val combineR = newRWithLN.map( (v) => (v._4, v)).partitionBy(new ExactPartitioner(numPivots)).cache()

    val combineRFinal = combineR.map(_._2).mapPartitions( (v) => {
      val vecMap:mutable.Map[Vector, List[(LabeledPoint, Double, Int)]] = mutable.HashMap.empty[Vector, List[(LabeledPoint, Double, Int)]].withDefaultValue(List())
      v.foreach( elem => {
        vecMap(elem._1) =  elem._2 ::: vecMap(elem._1)
        vecMap(elem._1) = vecMap(elem._1).distinct
        None
      })
      vecMap.map( elem => (elem._1, elem._2.sortBy(_._2).take(numNeighbours))).toIterator
    }
    ).cache()

    println("ERROR: Total count final is " + combineRFinal.count())

    result.union(combineRFinal.map(v => (v._1, v._2.flatMap(e => Some((e._1, e._2)))))).saveAsTextFile(pathName)

  }

  // Select pivots from a random sample groups based on max distance between each other
  protected def randomPivotSelect(rData: RDD[Vector]):Array[Vector] = {

    var total_dist = Double.MaxValue
    var computed_pivots = Array.empty[Vector]


    val pivot_distance = rData.mapPartitionsWithIndex { (idx, iter) =>
      val total_dist = iter.toSeq.combinations(2).foldLeft(0.0) { (dist, vecs) =>
        dist + Vectors.sqdist(vecs(0), vecs(1))
      }
      List((idx, total_dist)).toIterator
    }

    val partition_idx = pivot_distance.max()(Ordering.by(t => t._2))._1
    new PartitionPruningRDD[Vector](rData, (idx) => idx == partition_idx).collect()
  }

  // Compute pivots
  def computePivots(data: RDD[LabeledPoint], strategy: String = "random"): Array[Vector] = {


    val pData = data.takeSample(false, numPivots).map(_.features)
    return pData
//    return randomPivotSelect(data.context.parallelize(pData, 100))
  }

  override def copy(extra: ParamMap): KNearestNeighbourModel = {
    copyValues(new KNearestNeighbourModel(uid, distMetric, trainData).setParent(this.parent), extra)
  }

  override def toString: String = {
    s"KNearestNeighbour classes"
  }

}
