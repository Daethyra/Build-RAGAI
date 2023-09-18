# Automate the OP stack with `pinembed`

##### *A plugin for automating the retrieval of text embeddings from OpenAI and storing them in Pinecone.*

## Overview

This document outlines the recent updates made to a Python module designed for automating the retrieval of text embeddings from OpenAI and storing them in Pinecone. If you're new to this, think of text embeddings as numerical representations of textual data, and Pinecone as a storage service for these embeddings.

The key enhancements include the introduction of mechanisms to control the rate of API requests (rate-limiting) and improvements in the organization of the code (modularity). These changes aim to make the module robust and adaptable to API limitations, suitable for developers of all levels.

## Table of Contents

- [Automate the OP stack with `pinembed`](#automate-the-op-stack-with-pinembed)
      - [A plugin for automating the retrieval of text embeddings from OpenAI and storing them in Pinecone.](#a-plugin-for-automating-the-retrieval-of-text-embeddings-from-openai-and-storing-them-in-pinecone)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
      - [What Changed?](#what-changed)
      - [Intended Usage and Capabilities](#intended-usage-and-capabilities)
      - [Who Should Use This?](#who-should-use-this)
      - [What Can It Do?](#what-can-it-do)
    - [Introduction to Rate Limiting](#introduction-to-rate-limiting)
      - [What is Rate Limiting?](#what-is-rate-limiting)
      - [How is it Implemented?](#how-is-it-implemented)
    - [Modular Code Organization and Configuration](#modular-code-organization-and-configuration)
      - [What is Modular Code?](#what-is-modular-code)
    - [Glossary](#glossary)

#### What Changed?

The original implementation had a single class `PineconeHandler` that handled both OpenAI and Pinecone functionalities. The updated version introduces a separate `OpenAIHandler` class, isolating OpenAI-specific functionalities. Furthermore, environment variables are now managed better using `dotenv`.

### Intended Usage and Capabilities

#### Who Should Use This?

This module is designed to be user-friendly and robust, suitable for both beginners and experienced developers.

#### What Can It Do?

It can handle high volumes of text data, transform them into embeddings via the OpenAI API, and store them in Pinecone, all without violating any API limitations. It employs `asyncio` for efficient asynchronous operations.

### Introduction to Rate Limiting

#### What is Rate Limiting?

Rate limiting is the practice of controlling the number of requests sent to an API within a given time frame. This is important to ensure that we don't overwhelm the API service.

#### How is it Implemented?

The module now incorporates a `RateLimiter` class. This class is designed to manage the rate of API requests, enabling the module to align with the API limitations of both OpenAI (3,500 requests per minute) and Pinecone (100 vectors per upsert request).

### Modular Code Organization and Configuration

#### What is Modular Code?

Modular code means that the code is organized into separate sections or 'modules,' each handling a specific functionality. This makes the code easier to understand, test, and maintain.

### Glossary

- **API (Application Programming Interface)**: A set of rules that allows different software entities to communicate with each other.
- **Embedding**: A set of numerical values that represent the features of textual data.
- **Upsert**: A database operation that inserts rows into a database table if they do not already exist, or updates them if they do.
