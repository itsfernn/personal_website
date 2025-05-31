---
title: "{{ replace (path.Base .Dir) "-" " " | title }}"
date: {{ .Date }}
author: "Lukas Hofbauer"
github: "https://github.com/itsfernn/ml-daily-challenge/blob/main/{{ path.Base .Dir }}/{{ path.Base .Dir }}.ipynb"
cover:
    image: "cover.png"
---
