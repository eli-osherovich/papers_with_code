#!/bin/sh
isort --atomic .
yapf -ri .
