import networkx as nx
import matplotlib.pyplot as plt
from loguru import logger


def rdf_to_graph_visualization(graph):
    """
    Takes RDF graph as input and outputs a visually enhanced graph representation.

    :param graph: RDF graph parsed with rdflib
    """
    # Create a directed graph
    nx_graph = nx.DiGraph()

    # Add triples (subject, predicate, object) to the networkx graph
    for subj, pred, obj in graph:
        nx_graph.add_edge(subj, obj, label=pred)

    # Draw the graph with custom positioning and color
    pos = nx.spring_layout(
        nx_graph, seed=42)  # Force-directed layout for better visualization
    plt.figure(figsize=(12, 12))

    # Define custom node and edge properties
    node_color = 'skyblue'
    edge_color = 'gray'
    node_shape = 'o'  # Circular nodes
    node_size = 4000
    font_size = 12
    font_weight = 'bold'

    # Draw nodes with color and shape
    nx.draw(nx_graph, pos, with_labels=True, node_size=node_size,
            node_color=node_color, font_size=font_size, font_weight=font_weight,
            arrows=True, edge_color=edge_color, node_shape=node_shape)

    # Customize edge labels (predicates) with specific font size and color
    edge_labels = {(u, v): d['label'].split('/')[-1] for u, v, d in
                   nx_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels,
                                 font_color='darkgreen', font_size=10)

    # Highlight edges with a different color and style for clarity
    nx.draw_networkx_edges(nx_graph, pos, edge_color='orange', arrowstyle='->',
                           arrowsize=15, width=2)

    # Set title and formatting for the plot
    plt.title('Enhanced RDF Graph Representation', fontsize=16,
              fontweight='bold')

    # Streamlit integration: display the plot in Streamlit
    st.pyplot(plt)


def rdf_to_json_hierarchy(graph, subject, visited=None):
    if visited is None:
        visited = set()

    # Avoid circular references
    if subject in visited:
        return {"ref": str(subject)}

    visited.add(subject)

    subject_data = {"subject": str(subject), "properties": {}}

    for predicate, obj in graph.predicate_objects(subject):
        predicate_str = str(predicate)

        # Check if the object is another node or a literal
        if isinstance(obj, URIRef):
            # Recursively fetch the object
            subject_data["properties"].setdefault(predicate_str, []).append(
                rdf_to_json_hierarchy(graph, obj, visited)
            )
        else:
            subject_data["properties"].setdefault(predicate_str, []).append(
                str(obj))

    return subject_data


# Load the RDF file into an rdflib graph
def convert_rdf_to_json(graph):
    json_data = []
    for subject in graph.subjects():
        json_data.append(rdf_to_json_hierarchy(graph, subject))

    return json_data

def graph_summary(graph):
    """
    Summary of the graph, including the names, namespaces, unique subjects, and predicates.
    """
    namespaces = set()
    subjects = set()
    predicates = set()

    for subj, pred, obj in graph:
        namespaces.add(subj.split('#')[0])
        subjects.add(subj)
        predicates.add(pred)

    return {
        "Namespaces": list(namespaces),
        "Unique Subjects": list(subjects),
        "Unique Predicates": list(predicates)
    }


import streamlit as st
from rdflib import Graph, URIRef, RDF, RDFS, OWL
import os
import json


def create_ontology_summary_dict(ontology_graph, prefix_to_omit=None):
    def remove_prefix(uri):
        """Helper function to remove the given prefix from URIs."""
        if prefix_to_omit is None:
            return str(uri)
        return str(uri).lstrip(prefix_to_omit)
    # Extract ontology details
    ontology = set(ontology_graph.subjects(RDF.type, OWL.Ontology))
    # Extract classes
    classes = set(ontology_graph.subjects(RDF.type, OWL.Class))
    # Extract object properties
    object_properties = set(ontology_graph.subjects(RDF.type, OWL.ObjectProperty))
    # Extract data properties and their hierarchy
    data_properties = set(ontology_graph.subjects(RDF.type, OWL.DatatypeProperty))
    data_property_hierarchy = {}
    for dp in data_properties:
        sub_properties = list(ontology_graph.objects(dp, RDFS.subPropertyOf))
        data_property_hierarchy[remove_prefix(dp)] = [
            remove_prefix(sub) for sub in sub_properties
        ] if sub_properties else None
    # Extract annotation properties
    annotation_properties = set(ontology_graph.subjects(RDF.type, OWL.AnnotationProperty))
    # Summary of the RDF content
    ontology_summary = {
        "Ontology": [remove_prefix(onto) for onto in ontology],
        "Classes": [remove_prefix(cls) for cls in classes],
        "Object Properties": [remove_prefix(prop) for prop in
                              object_properties],
        "Data Properties": {
            "Properties": [remove_prefix(prop) for prop in data_properties],
            "Hierarchy": data_property_hierarchy
        },
        "Annotation Properties": [remove_prefix(prop) for prop in
                                  annotation_properties]
    }

    return ontology_summary


def display_ontology_summary(ontology_summary):
    st.subheader("Ontology")
    if ontology_summary["Ontology"]:
        for onto in ontology_summary["Ontology"]:
            st.write(f" - {onto}")
    else:
        st.write("No ontology information available.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Classes")
    if ontology_summary["Classes"]:
        st.json(ontology_summary["Classes"], expanded=False)
    else:
        st.write("No classes available.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Object Properties")
    if ontology_summary["Object Properties"]:
        st.json(ontology_summary["Object Properties"], expanded=False)
    else:
        st.write("No object properties available.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Data Properties")
    if ontology_summary["Data Properties"]:
        st.json(ontology_summary["Data Properties"], expanded=False)
    else:
        st.write("No data properties available.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Annotation Properties")
    if ontology_summary["Annotation Properties"]:
        st.json(ontology_summary["Annotation Properties"], expanded=False)
    else:
        st.write("No annotation properties available.")


def get_subclasses(graph, parent_class):
    """
    Recursive function to get all subclasses of a given parent class.
    """
    subclasses = {}

    # Find all subclasses of the parent_class
    for subclass in graph.subjects(RDFS.subClassOf, parent_class):
        # Recurse to find subclasses of this subclass
        subclasses[subclass] = get_subclasses(graph, subclass)

    return subclasses


def display_class_hierarchy(graph, omit_prefix=None):
    """
    Build a class hierarchy from the RDF graph as a JSON-like dictionary.
    """
    # Start by finding all top-level classes (which have no superclass)
    hierarchy = {}

    def omit(iri):
        if omit_prefix:
            return iri.lstrip(omit_prefix)
        else:
            return iri
    for owl_class in graph.subjects(RDF.type, OWL.Class):
        # Check if this class has no superclass (i.e., top-level)
        if not list(graph.objects(owl_class, RDFS.subClassOf)):
            hierarchy[omit(owl_class)] = get_subclasses(graph, owl_class)
    st.json(hierarchy, expanded=False)


@st.cache_data
def read_raw_file(file_path):
    with open(file_path, 'r') as file:
        logger.info(f"Reading raw RDF content from {file_path}")
        return file.read()


def display_raw_rdf(rdf_path):
    """
    Display the raw RDF content from a file
     when the `Show raw RDF` button is clicked.
    """
    st.code(read_raw_file(rdf_path), language="xml")
