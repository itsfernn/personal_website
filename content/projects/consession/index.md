---
date: '2025-04-22T12:05:03+09:00'
draft: false
title: 'Consession'
github: "https://github.com/itsfernn/consession"
endless: 'A tasty session manager for tmux'
cover:
    image: "cover.gif"
    alt:  "Consession logo"
---

I love working in the terminal, there's nothing quite like it. Once you’ve set up your workflow, navigating your system becomes second nature. Flying through the filesystem, quickly editing files, batch-moving specific file types, writing quick bash scripts, or chaining commands together with pipes — it’s all possible without ever lifting your hands from the keyboard.

Learning to use the terminal effectively might be the highest ROI you can make as a developer. Every new tool you master becomes a force multiplier, letting you work faster and more efficiently.

Today, I’m excited to announce a new tool I’ve created that fits right into that mindset: Consession — and it might just be one of my favorites.

## TMUX

To understand consession you need to first understand TMUX. Let me pitch it to you.

If you spend a lot of time in the terminal, you've probably run into this: a bunch of open terminal windows, one of them running an important process... and then you accidentally close it. Gone.

TMUX solves this problem by decoupling terminal sessions from the terminal window. Your sessions continue running in the background, and you simply “attach” to them. That means you can close and reopen your terminal window without losing your work — no more accidental kills. It’s a game-changer.

But TMUX has its own rough edges. Managing sessions, creating, renaming, switching, deleting can be tedious. That’s where Consession comes in.


## Consession

Consession is a terminal UI for managing your TMUX sessions. It leverages fzf to give you a clean, interactive overview of all running sessions. With intuitive keybindings, you can rename or delete sessions on the fly. Fuzzy search makes switching between sessions a breeze.

But the killer feature is the Zoxide integration.

If your search doesn’t match an existing session, Consession opens a view of your most visited directories via Zoxide. You can instantly start a new TMUX session in any of them.

It makes jumping around your system effortless. Want to tweak your nvim config? Just launch Consession and start typing “nvim.” No setup needed — Zoxide already knows where you’ve been.

Consession brings sanity to TMUX session management and makes terminal-based workflows feel even smoother. Give it a try and let me know what you think!
