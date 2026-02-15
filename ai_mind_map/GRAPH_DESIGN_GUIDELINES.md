# AI Learning Path Graph - Design Decisions & Guidelines

This document explains our design choices for the AI learning path visualization and provides guidelines for future updates.

---

## Core Philosophy: Foundation Flow

**Goal:** Show the essential learning progression without visual clutter.

**Approach:** Keep only conceptual prerequisites and foundational relationships. Avoid tool-to-technique connections and overly granular dependencies.

---

## Connection Types & Rules

### ✅ KEEP These Connections

#### 1. **Hierarchical Structure (Parent → Child)**
All main topics expand into subtopics:
```
AI → Deep Learning → Neural Networks, CNNs, RNNs, Transformers
AI → NLP → Text Processing, Embeddings, Language Models
```

**Rule:** Every Level 1 node connects to its Level 2 children.

#### 2. **Foundation Prerequisites (Concept → Concept)**
Show the natural learning progression:

**Essential Chain:**
```
Mathematics → Machine Learning → Deep Learning
```

**Rationale:**
- You need math (linear algebra, calculus, probability) to understand ML
- You need classical ML concepts before tackling deep learning
- This is the universally recommended learning path

#### 3. **Enablement Relationships (Foundation → Application)**
Show how foundational concepts enable entire domains:

```
Deep Learning → Computer Vision
Deep Learning → NLP  
Deep Learning → Reinforcement Learning
```

**Rationale:**
- Modern CV is primarily deep learning-based (CNNs, Vision Transformers)
- Modern NLP is primarily deep learning-based (Transformers, LLMs)
- Modern RL uses deep learning (DQN, policy gradients)
- These show "you need DL to do modern work in these fields"

**Total Connections:** 5 cross-domain links only

---

### ❌ REMOVE These Connections

#### 1. **Tool-to-Concept Links**
**Examples we removed:**
- ~~PyTorch → Neural Networks~~
- ~~TensorFlow → Deep Learning~~
- ~~Hugging Face → Transformers~~

**Rationale:** 
- Tools are implementation details, not prerequisites
- You can learn concepts independently of specific tools
- Creates visual clutter without educational value
- Tools change frequently; concepts remain stable

#### 2. **Granular Technique Dependencies**
**Examples we removed:**
- ~~Transformers → NLP~~ (already covered by DL → NLP)
- ~~CNNs → Computer Vision~~ (already covered by DL → CV)
- ~~Transformers → Language Models~~ (implied in hierarchy)
- ~~Calculus → Optimization~~ (too granular)
- ~~Neural Networks → Supervised Learning~~ (backwards relationship)

**Rationale:**
- These are implied by the hierarchy or broader connections
- Too specific; creates spaghetti graph
- If DL enables NLP, and Transformers are part of DL, the connection is implied

#### 3. **Bidirectional or Circular Dependencies**
**Examples we removed:**
- ~~Data Engineering → Machine Learning → Data Engineering~~
- ~~Tools → Deep Learning~~ (tools don't enable concepts)
- ~~Ethics → AI Core~~ (ethics applies to all, not a prerequisite)

**Rationale:**
- Avoid circular thinking
- Keep unidirectional learning flow
- Ethics and tools are supporting topics, not in the critical path

#### 4. **Cross-Application Domain Links**
**What we DON'T connect:**
- Computer Vision ↔ NLP
- NLP ↔ Reinforcement Learning
- CV ↔ RL

**Rationale:**
- These are parallel specializations, not sequential
- You can learn them independently
- Connections would suggest false prerequisites

---

## Node Organization Guidelines

### Level 0: Central Node
- **Single node:** "Artificial Intelligence"
- This is the root of everything

### Level 1: Main Domains (10 nodes)
Categories representing major areas of study:

1. **Mathematics** - Foundation theory
2. **Machine Learning** - Core algorithms  
3. **Deep Learning** - Neural network techniques
4. **Computer Vision** - Image/video AI
5. **NLP** - Language and text AI
6. **Reinforcement Learning** - Agent-based learning
7. **Tools & Frameworks** - Implementation tools
8. **Data Engineering** - Data handling
9. **Research & Advanced** - Cutting-edge topics
10. **AI Ethics** - Responsible AI

**Color Coding:**
- Mathematics: Blue (#3b82f6) - Foundation
- ML: Purple (#8b5cf6) - Core learning
- DL: Pink (#ec4899) - Advanced core
- CV: Green (#10b981) - Application domain
- NLP: Orange (#f59e0b) - Application domain
- RL: Teal (#14b8a6) - Application domain
- Tools: Indigo (#6366f1) - Practical
- Research: Purple (#a855f7) - Advanced
- Ethics: Gray (#64748b) - Cross-cutting

### Level 2: Subtopics (49 nodes)
Specific techniques, methods, and concepts within each domain.

**Guidelines for Level 2:**
- 4-6 nodes per Level 1 domain
- Most important/fundamental concepts only
- Practical and commonly used techniques
- Avoid overly specific or niche topics

---

## Adding New Nodes: Decision Framework

### When Adding a New Node, Ask:

#### 1. **Is it fundamental enough?**
- ✅ Yes: "Graph Neural Networks" (major DL architecture)
- ❌ No: "YOLO v8" (too specific, covered under Object Detection)

#### 2. **Where does it fit in the hierarchy?**
- Level 0: Never (only one AI root)
- Level 1: Is it a major domain? (rare to add)
- Level 2: Is it a key technique within a domain? (most additions)

#### 3. **What connections should it have?**
- **Always:** Parent connection (e.g., Graph Neural Networks → Deep Learning)
- **Sometimes:** Prerequisite if it's a foundation (e.g., Math → Statistics)
- **Never:** Tool connections, granular dependencies, or cross-domain unless absolutely essential

#### 4. **Does it overlap with existing nodes?**
- Avoid duplication
- If similar node exists, consider expanding its info instead of adding new node

---

## Examples: Adding New Content

### Example 1: Adding "Self-Supervised Learning"
**Decision:**
- Fundamental? ✅ Yes (major learning paradigm)
- Level: 2 (subtopic of ML)
- Parent: Machine Learning
- Extra connections: None (it's a parallel concept to Supervised/Unsupervised)

**Implementation:**
```javascript
{ id: 60, name: "Self-Supervised Learning", level: 2, category: "ml" }
// Connection:
{ source: 2, target: 60 }  // ML → Self-Supervised Learning
```

### Example 2: Adding "Edge AI"
**Decision:**
- Fundamental? ⚠️ Borderline (deployment topic)
- Level: Could be under Tools or Research
- Better approach: Add to MLOps description instead of new node
- Avoid clutter for specialized deployment method

**Implementation:**
Update node 44 (MLOps Tools) info to mention Edge AI.

### Example 3: Adding "Reinforcement Learning from Human Feedback (RLHF)"
**Decision:**
- Already exists! (Node 40)
- If adding detail, update the nodeInfo for id 40

---

## Future Expansion Checklist

When adding multiple nodes or restructuring:

1. **Review the 5 core connections** - Do they still make sense?
```javascript
{ source: 1, target: 2 },   // Math → ML
{ source: 2, target: 3 },   // ML → DL
{ source: 3, target: 4 },   // DL → CV
{ source: 3, target: 5 },   // DL → NLP
{ source: 3, target: 6 }    // DL → RL
```

2. **Maintain balance** - Each Level 1 should have 4-6 Level 2 nodes

3. **Test visual clarity** - Open the graph and drag nodes around
   - Does it feel cluttered?
   - Can you see clear learning paths?
   - Are connections crossing unnecessarily?

4. **Update color scheme** if adding new Level 1 domains

5. **Update the legend** if new categories added

---

## Node Info Structure

When adding nodeInfo for new nodes:

```javascript
nodeId: {
  title: "Node Name",
  status: "current|planned|foundational",  // Learning status
  description: "1-2 sentence overview",
  details: ["Bullet point 1", "Bullet point 2", "..."],  // 3-5 items
  keyTopics: ["Topic 1", "Topic 2", "..."],  // What to learn
  links: [
    { text: "Display Text", url: "https://github.com/.../README.md" }
  ]
}
```

**Status Guidelines:**
- `current`: You're actively learning/working on this
- `planned`: Future learning goals
- `foundational`: Essential prerequisites everyone needs

---

## Visual Design Considerations

### Force Simulation Parameters
Current settings work well for 60 nodes:

```javascript
.force('link', d3.forceLink(data.links).id(d => d.id).distance(d => {
  return d.target.level === 2 ? 80 : 150;  // Shorter for L2, longer for L1
}))
.force('charge', d3.forceManyBody().strength(d => {
  if (d.level === 0) return -2000;  // Strong repulsion for center
  if (d.level === 1) return -1000;  // Medium for L1
  return -500;  // Light for L2
}))
```

**If adding 10+ nodes:** Consider adjusting charge strength to prevent overcrowding.

### Text Readability
Keep font weight light (200) with white stroke:
```css
stroke: white;
stroke-width: 3px;
font-weight: 200;
```

---

## Philosophy Summary

**Remember:** The goal is a learning roadmap, not a complete reference.

- **Clarity over completeness** - Better to have 60 well-chosen nodes than 200 cluttered ones
- **Concepts over tools** - PyTorch may change, but neural networks remain
- **Learning flow over accuracy** - Show the path, not every connection
- **Visual simplicity** - If it looks messy, simplify

---

## Version History

**v1.0 (Current):** Foundation Flow
- 60 nodes (1 L0, 10 L1, 49 L2)
- 5 cross-domain connections
- Clean visual layout
- Focus: Essential learning prerequisites

**Previous:** Full Connected (deprecated)
- 17 cross-domain connections
- Included tool-to-technique links
- Too cluttered for effective use

---

## Contact/Questions

When in doubt:
1. Favor simplicity
2. Ask: "Does this connection show a true prerequisite?"
3. Test visually - if it looks messy, it probably is
4. Remember: You can always add node info without adding new nodes

Keep the graph clean, keep it focused, keep it useful. 🎯
