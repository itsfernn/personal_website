---
title: "{{ replace (path.Base .Dir) "-" " " | title }}"
date: {{ .Date }}
author: "Lukas Hofbauer"
github: "https://github.com/itsfernn/{{ path.Base .Dir }}"
cover:
    image: "cover.png"
---
