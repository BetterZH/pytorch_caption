#! /usr/bin/env python
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Modified by Alexander Rush, 2007

""" Generate beam search visualization.
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import argparse
import os
import json
import shutil
from string import Template

import networkx as nx
from networkx.readwrite import json_graph

PARSER = argparse.ArgumentParser(
    description="Generate beam search visualizations")
PARSER.add_argument(
    "-d", "--data", type=str, required=True,
    help="path to the beam search data file")
PARSER.add_argument(
    "-o", "--output_dir", type=str, required=True,
    help="path to the output directory")
ARGS = PARSER.parse_args()


HTML_TEMPLATE = Template("""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Beam Search</title>
    <link rel="stylesheet" type="text/css" href="tree.css">
    <script src="http://d3js.org/d3.v3.min.js"></script>
  </head>
  <body>
    <a href="$URL_PREV">PREV</a>
    <a href="$URL_NEXT">NEXT</a>
    <img src="$IMG_SRC" width='400px'/>
    <h3>$SENT</h3>
    <script>
      var treeData = $DATA
    </script>
    <script src="tree.js"></script>
  </body>
</html>""")


def _add_graph_level(graph, level, parent_ids, names, scores):
  """Adds a levelto the passed graph"""
  for i, parent_id in enumerate(parent_ids):
    new_node = (level, i)
    parent_node = (level - 1, parent_id)
    score_str = '%.3f' % float(scores[i]) if scores[i] is not None else '-inf'
    graph.add_node(new_node)
    graph.node[new_node]["name"] = names[i]
    graph.node[new_node]["score"] = score_str
    graph.node[new_node]["size"] = 100
    # Add an edge to the parent
    graph.add_edge(parent_node, new_node)

def create_graph(predicted_ids, parent_ids, scores, vocab=None):
  def get_node_name(pred):
    return vocab[str(pred)] if vocab else pred

  seq_length = len(predicted_ids) #.shape[0]
  graph = nx.DiGraph()
  for level in range(seq_length):
    names = [get_node_name(pred) for pred in predicted_ids[level]]
    _add_graph_level(graph, level + 1, parent_ids[level], names, scores[level])
  graph.node[(0, 0)]["name"] = "START"
  return graph


def main():
  beam_data = json.load(open(ARGS.data, 'r'))

  # Optionally load vocabulary data
  vocab = beam_data['vocab']
  vocab['0'] = '<EOS>'
  # vocab = None

  if not os.path.exists(ARGS.output_dir):
    os.makedirs(ARGS.output_dir)

  path_base = os.path.dirname(os.path.realpath(__file__))

  # Copy required files
  shutil.copy2(path_base+"/beam_search_viz/tree.css", ARGS.output_dir)
  shutil.copy2(path_base+"/beam_search_viz/tree.js", ARGS.output_dir)

  len_ids = len(beam_data["predicted_ids"])
  for idx in range(len(beam_data["predicted_ids"])):
    predicted_ids = beam_data["predicted_ids"][idx]
    parent_ids = beam_data["beam_parent_ids"][idx]
    scores = beam_data["scores"][idx]
    image_id = beam_data["ids"][idx]
    sent = beam_data["sents"][idx]

    graph = create_graph(
        predicted_ids=predicted_ids,
        parent_ids=parent_ids,
        scores=scores,
        vocab=vocab)

    json_str = json.dumps(
        json_graph.tree_data(graph, (0, 0)),
        ensure_ascii=True)


    url_prev = "{:06d}.html".format((idx - 1 + len_ids) % len_ids)
    url_next = "{:06d}.html".format((idx + 1)%len_ids)

    img_path = "../testb_imgs/"+image_id+".jpg"
    html_str = HTML_TEMPLATE.substitute(DATA=json_str,SENT=sent,IMG_SRC=img_path,URL_PREV=url_prev,URL_NEXT=url_next)
    output_path = os.path.join(ARGS.output_dir, "{:06d}.html".format(idx))
    with open(output_path, "w") as file:
      file.write(html_str)
    print(output_path)


if __name__ == "__main__":
  main()
