/*
 * Authors: Siddharth D Jaiswal, Dr. Yogesh Simmhan: 22/9/2017
 * @see <a href="http://www.dream-lab.in/">DREAM:Lab</a>
 * <p>
 * Copyright 2014 DREAM:Lab, Indian Institute of Science, Bangalore
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain
 * a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
* */

package in.dream_lab.goffish.sample;
import in.dream_lab.goffish.api.*;

import com.google.common.collect.Iterables;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.*;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.*;


public class DynamicSgr extends AbstractSubgraphComputation<LongWritable, LongWritable, LongWritable, LongDoubleWritable, LongWritable, LongWritable, LongWritable> implements ISubgraphWrapup{

    private static final Object lock = new Object(); //need lock to read from file 
    private static Map<Long, Double> RankMap = null ; //will need this to read values from previous snapshots
    private static final String path = "/user/humus/FBTempIn4/"; //path to read previous file from
    private static final String HDFSPath = "hdfs://orion-00:19000"; //hdfs path where file is stored

    private static final double ALPHA = 0.85;  //standard default
    private static final double EPSILON = 1e-4; //threshold for convergence
    private double defaultPR; //set a default pr value of 1/|V| for all vertices to start with
    private boolean converged =false;

    private Map<Long, Double> ranks = new HashMap<>(); //pagerank value per vertex stored here
    private Map<Long, Double> localSums = new HashMap<>(); //local sums to converge to pr values
    private Map<Long, Double> remoteSums = new HashMap<>(); //for remote vertices, does same job as above
    private Map<Long, Integer> outDegrees = new HashMap<>(); //outdegree of every vertex, necessary for PR

    @Override
    public void compute(Iterable<IMessage<LongWritable, LongDoubleWritable>> iMessages) throws IOException {

        if(getSuperstep()==0) {
            // Local Page Rank for first snapshot but fill in with previous values if further snapshots

            Map<Long, Double> localRankMap = getRankMap(getSubgraph().getSubgraphId().get() >> 24);
            int globalcount = 0;

            for(IMessage<LongWritable, LongDoubleWritable> message : iMessages)
                globalcount += message.getMessage().getVertexId();

            double G_by_SG = ((double) globalcount/(getSubgraph().getLocalVertexCount()));

            for(IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getVertices()) {
                long vertexId = vertex.getVertexId().get();
                double rank = 0.0;
                if (localRankMap.get(vertexId) != null) {
                    rank = localRankMap.get(vertexId) * G_by_SG;
                    ranks.put(vertexId,rank);
                }

                else {
                    LPRCompute(); 
                    break;
                }
            }


            sendToAll(new LongDoubleWritable(getSubgraph().getLocalVertexCount(), 0.0));
           // System.out.println(getSuperstep() + ": SGid: " + getSubgraph().getSubgraphId() + " : Vn: " + getSubgraph().getLocalVertexCount()); //remove later
            //SGid is id of subgraph and Vn is no. of vertices in that subgraph
        }

        else if (getSuperstep() == 1) {
            // Call this after local convergence is achieved and then multiply with SG/G to get global PR

            double LPRSgr = 0.0 ;
            int globalVertexCount = 0;

            for(IMessage<LongWritable, LongDoubleWritable> message : iMessages)
                globalVertexCount += message.getMessage().getVertexId();

            double SG_by_G = ((double) getSubgraph().getLocalVertexCount())/globalVertexCount;
            // Global default PR
            defaultPR = 1/globalVertexCount;

            for(Long vertex : ranks.keySet()) {
                ranks.put(vertex, ranks.get(vertex) * SG_by_G);
            }

            //This loop can be removed. Doesn't contribute to algo itself
            for (IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getLocalVertices()) {
                LPRSgr = Math.max(LPRSgr, ranks.get(vertex.getVertexId().get()));
            }
            //System.out.println(getSuperstep() + ": The SGR value after LPR*SG/G compute " + LPRSgr ); //remove later
            GPRCompute();
        }

        else {
            //If Global convergence not seen yet, call this to converge globally
            for (IMessage<LongWritable, LongDoubleWritable> message : iMessages) {
                if (message.getMessage().getVertexId() == -1)
                    // global NOT convergence message seen
                    converged = false;
                else
                    // PR message seen. Add up remote message sums with prior local sums.
                    localSums.put(message.getMessage().getVertexId(),
                            localSums.get(message.getMessage().getVertexId()) + message.getMessage().getSums());

            }

            if(!converged)
                GPRCompute();
            else
                voteToHalt();
        }
    }

    public void LPRCompute() {
        final double EPSILON_LPR = 0.05; //Local convergence
        double LPRmax = 0.0;

        Map<Long, Integer> localOutDegrees = new HashMap<>();

        //Use the next statement only if first snapshot. For all others, read from file and populate.
        double localDefaultPR = 1.0D / getSubgraph().getLocalVertexCount(); //1 for sg having 1 v

        // initialize map for local ranks and local outDegrees, and sums from local neighbors
        for(IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getVertices()) {
            if(!vertex.isRemote()) {
                ranks.put(vertex.getVertexId().get(), localDefaultPR); //this is 1 for subgraphs with 1 vertex
                localSums.put(vertex.getVertexId().get(), 0.0);
                outDegrees.put(vertex.getVertexId().get(), Iterables.size(vertex.getOutEdges()));

                // TODO: we can replace this once local out degree count is available in API
                int localOutDegree = 0;
                for (IEdge<LongWritable, LongWritable, LongWritable> edge : vertex.getOutEdges()) {
                    if (!getSubgraph().getVertexById(edge.getSinkVertexId()).isRemote())
                        localOutDegree++;
                }
                localOutDegrees.put(vertex.getVertexId().get(), localOutDegree);
                //System.out.println(getSuperstep() + ": Vertex Id " + vertex.getVertexId() + " has localoutdegree " + localOutDegree); //remove later
            }
            else
                remoteSums.put(vertex.getVertexId().get(), 0.0);
        }

        if(getSubgraph().getLocalVertexCount() == 1){
            for(IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getVertices()) {
                ranks.put(vertex.getVertexId().get(), 1.0D);
            }
        }

        else {
            // ====================================
            // Do local PR till convergence
            do {

                // update sums for local and remote vertices, O(E+LV+RV)
                for (IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getLocalVertices()) {
                    double weight = ranks.get(vertex.getVertexId().get()) / localOutDegrees.get(vertex.getVertexId().get());
                    for (IEdge<LongWritable, LongWritable, LongWritable> edge : vertex.getOutEdges()) {
                        if (!getSubgraph().getVertexById(edge.getSinkVertexId()).isRemote())
                            // for local edges only
                            localSums.put(edge.getSinkVertexId().get(),
                                    localSums.get(edge.getSinkVertexId().get()) + weight);
                    }
                }
                converged = true;
                // update local PR values
                for (Long vertexId : localSums.keySet()) {
                    double pr_old = ranks.get(vertexId);
                    ranks.put(vertexId, ALPHA * localSums.get(vertexId) + (1.0 - ALPHA) * localDefaultPR);
                    if (Math.abs(pr_old - ranks.get(vertexId)) > EPSILON_LPR)
                        converged = false;
                    localSums.put(vertexId, 0.0);
                }
            } while (!converged);
        }
        for (IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getLocalVertices())
            LPRmax = Math.max(LPRmax, ranks.get(vertex.getVertexId().get()));
        // for global convergence
        converged =false;
        //System.out.println(getSuperstep()+ ": The SGR value from LPR compute for subgraph" + getSubgraph().getSubgraphId() +" is"+ LPRmax ); //remove later
    }

    public void GPRCompute(){
        // ====================================
        // update local PR values, local/remote sums, local convergence, O(LV)
        double LPRmax = 0.0;

        if(getSuperstep()!=1) {
            double abssum = 0.0;
            double absavg = 0.0;
            converged = true;
            for (Long vertexId : localSums.keySet()) {
                double pr_old = ranks.get(vertexId);
                ranks.put(vertexId, ALPHA * localSums.get(vertexId) + (1.0 - ALPHA) * defaultPR);
                if (Math.abs(pr_old - ranks.get(vertexId)) > EPSILON) {
                    abssum = abssum + Math.abs(pr_old - ranks.get(vertexId));
                    converged = false;
                }
                localSums.put(vertexId, 0.0); // reset local sum for this vertex
                LPRmax = Math.max(LPRmax,ranks.get(vertexId));
            }
//            absavg = abssum/(double)(getSubgraph().getLocalVertexCount());
            //System.out.println(getSuperstep() + " : Residual Sum: " + abssum);
            //System.out.println(getSuperstep() + ": The PR value from LPR compute " + LPRmax );
        }

        // if not converged, broadcast special message with NOT converged flag, O(SG)
        if(!converged) {
            sendToAll(new LongDoubleWritable(-1,0));
        }

        // ====================================
        // ASSERT: Local sums and remote sums in Map are 0.0. Local PR value is updated.

        // ====================================
        // update sums for local and remote vertices, O(E+LV+RV)
        for(IVertex<LongWritable, LongWritable, LongWritable, LongWritable> vertex : getSubgraph().getLocalVertices()){
            double weight = ranks.get(vertex.getVertexId().get()) / outDegrees.get(vertex.getVertexId().get());
            for(IEdge<LongWritable, LongWritable, LongWritable> edge : vertex.getOutEdges()) {
                if (getSubgraph().getVertexById(edge.getSinkVertexId()).isRemote()) {
                    remoteSums.put(edge.getSinkVertexId().get(),
                            remoteSums.get(edge.getSinkVertexId().get()) + weight);
                }
                else {
                    localSums.put(edge.getSinkVertexId().get(),
                            localSums.get(edge.getSinkVertexId().get()) + weight);
                }
            }
        }

        // Send PR message to all remote neighbors, O(RV)
        for (IRemoteVertex<LongWritable, LongWritable, LongWritable, LongWritable, LongWritable> remote : getSubgraph()
                .getRemoteVertices()) {
            long remoteId = remote.getVertexId().get();
            sendMessage(remote.getSubgraphId(), new LongDoubleWritable(remoteId, remoteSums.get(remoteId)));
            remoteSums.put(remoteId, 0.0); // Init remote sum once message sent
        }
    }

    @Override
    public void wrapup() throws IOException {
        for(Long vertexId : ranks.keySet())
            System.out.println(vertexId + " " + ranks.get(vertexId));
    }


    private static Map<Long, Double> getRankMap(long partitionId) {
        synchronized (lock) {
            if (RankMap == null) {
                try {
                    Configuration conf = new Configuration();
                    conf.set("fs.default.name", HDFSPath);
                    FileSystem dfs = FileSystem.get(conf);
                    RankMap = new HashMap<>();
                    FSDataInputStream in = dfs.open(new Path(path + String.format("%05d", partitionId)));
                    Scanner sc = new Scanner(in);
                    while (sc.hasNextLong()) {
                        long vertexId = sc.nextLong();
                        double rank = sc.nextDouble();
                        RankMap.put(vertexId, rank);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            return RankMap;
        }
    }


}


/*class LongDoubleWritable implements Writable {

    private long vertexId;
    private double rank;

    public LongDoubleWritable(){}

    LongDoubleWritable(long vertexId, double rank) {
        this.vertexId = vertexId;
        this.rank = rank;
    }

    public long getVertexId() {
        return vertexId;
    }

    public double getRank() {
        return rank;
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        dataOutput.writeLong(vertexId);
        dataOutput.writeDouble(rank);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        vertexId = dataInput.readLong();
        rank = dataInput.readDouble();
    }
}*/