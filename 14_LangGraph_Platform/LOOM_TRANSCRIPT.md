# LangGraph Studio Demo - Video Transcript

**[00:00 - 00:15] Opening**

Hey everyone! Today I'm going to walk you through LangGraph Studio and show you how we can use it to visualize, debug, and test agentic workflows. This is part of Session 14 where we're building and serving agentic graphs with LangGraph Platform.

**[00:15 - 00:45] Project Overview**

So what we have here is a project with two different agent implementations. The first is a simple agent that uses tools conditionally, and the second is an agent with a helpfulness evaluation layer. Both are served locally through LangGraph's development server, and we can interact with them through LangGraph Studio. Let me show you what makes this powerful.

**[00:45 - 01:15] Starting the Server**

First, I have my local LangGraph server running on port 2024. You start this with the command `uv run langgraph dev`. Once that's up, I can access LangGraph Studio by going to the URL: smith.langchain.com/studio with the baseUrl parameter pointing to localhost:2024. 

What's great about Studio is that it connects directly to your local development server, so you can test your graphs in real-time without deploying anything to production.

**[01:15 - 02:00] Visualizing the Graphs**

Now let me show you the two different agents we have configured. In the langgraph.json file, we've defined two assistants. The first one, called "agent", uses the simple_agent graph. This is a straightforward tool-calling agent that executes tools when needed and terminates when done.

The second assistant, "agent_helpful", uses the agent_with_helpfulness graph. This is more sophisticated - after the agent generates a response, it goes through a helpfulness evaluation node. If the response isn't helpful enough, it loops back to the agent to try again. There's also a safety limit of 10 messages to prevent infinite loops.

**[02:00 - 02:30] Testing with Interrupts**

One of the most powerful features I want to show you is the interrupt functionality. In Studio, I can set interrupts before or after any node. For example, if I set an interrupt_before on the "helpfulness" node, the execution pauses right before the helpfulness check runs. This lets me inspect the agent's response and even modify it before it gets evaluated.

If I set an interrupt_after on the "agent" node, I can see what the agent produced and modify the tool calls before they execute. This is incredibly useful for debugging and understanding how your agent makes decisions.

**[02:30 - 03:00] Practical Testing**

Let me run a quick test. I'll send a query like "What is the MuonClip optimizer and what paper did it first appear in?" The agent will use the search tools to find this information. In Studio, I can watch each node execute in real-time, see the messages being passed between nodes, and observe how the graph routes between the agent, action, and helpfulness nodes.

**[03:00 - 03:30] Why This Is Useful**

So why is this useful? Three main reasons:

First, visual debugging - instead of reading logs, you see your graph execute node by node. You can spot routing issues, unexpected tool calls, or problematic responses immediately.

Second, interactive testing - the interrupt functionality lets you pause execution, modify state, and continue. This is game-changing for testing edge cases and understanding complex agent behaviors.

Third, rapid iteration - you can test changes to your graph instantly without redeploying. Modify your code, and Studio picks up the changes through the local dev server.

**[03:30 - 03:45] Closing**

LangGraph Studio transforms agent development from a black box into a transparent, interactive experience. You can see exactly what your agents are doing, test different scenarios with interrupts, and iterate much faster than traditional debugging methods.

Thanks for watching, and happy building!

---

**Total Duration: ~3:45 minutes**

**Key Topics Covered:**
- LangGraph Studio setup and connection to local dev server
- Two agent architectures: simple vs. helpfulness-evaluated
- Visual graph execution and node-by-node streaming
- Interrupt functionality (before/after nodes)
- Practical benefits: debugging, testing, and iteration

**Demo Flow Suggestions:**
1. Show terminal with `langgraph dev` running
2. Open LangGraph Studio in browser
3. Select and visualize both assistant types
4. Run a query with the simple agent
5. Run a query with the helpfulness agent
6. Demonstrate setting an interrupt and modifying state
7. Show the difference in execution paths between both agents


