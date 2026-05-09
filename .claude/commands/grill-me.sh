#!/bin/bash
# Custom grill-me command
# Interview about a plan until reaching shared understanding

cat <<'EOF'
Interview mode: grill-me

I will interview you relentlessly about every aspect of your plan until we reach shared understanding. I'll walk down each branch of the design tree, resolving dependencies between decisions one by one.

If a question can be answered by exploring the codebase, I will explore the codebase instead.

For each question, I will provide my recommended answer.

Ready to be grilled. What's your plan?
EOF
