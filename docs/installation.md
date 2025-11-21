# 📦 Installing Python Package via Azure DevOps Artifacts

This guide explains how to securely install the `agent_workflow` Python package from a **private Azure DevOps Artifacts feed** using a Personal Access Token (PAT).

---

##  Step 1: Generate a Personal Access Token (PAT)

The PAT grants **read-only access** to your package feed.

1. Navigate to **Azure DevOps**
2. Click your **User Icon** (top-right corner)
3. Select **Personal access tokens**
4. Click **New Token**
5. Under **Scopes**, select:
   - **Packaging**
   - **Read**
6. Click **Create**
7. Copy the generated token — you will **not** be able to view it again

>  Treat your PAT like a password. Do not share or store it in source code. 
> We will refer to it as: `<YOUR_PAT_HERE>`

---

## Step 2: Authenticate pip (`.netrc` Method)

This securely stores your PAT so pip can authenticate **automatically**, without exposing it in your terminal history.

### 🔹 For Windows Users

```sh
cd %USERPROFILE%
notepad .netrc
```

Paste the following into the `.netrc` file:

```
machine pkgs.dev.azure.com
    login azure
    password <YOUR_PAT_HERE>
```

---

### 🔹 For Ubuntu/Linux Users

```sh
nano ~/.netrc
```

Paste the same content:

```
machine pkgs.dev.azure.com
    login azure
    password <YOUR_PAT_HERE>
```

Optional but recommended (restricts access):

```sh
chmod 600 ~/.netrc
```

---

##  Step 3: Install the Package

Run the following command to install your private package:

```sh
pip install --pre agent-workflow-framework             --index-url https://pkgs.dev.azure.com/AIPlanetFramework/agent_framework/_packaging/FEED/pypi/simple/
```

- `--index-url` → Your private Azure Artifacts feed 
- `--extra-index-url` → Public PyPI (fallback for dependencies)

---

##  Benefits of This Authentication Method

| Benefit | Description |
|---------|-------------|
|  No wheel file sharing | No manual transfer of `.whl` files |
|  Secure authentication | PAT stored in `.netrc`, *not* exposed in commands |
|  Professional workflow | Same as any public pip package installation |
|  Zero code changes | No storing tokens in scripts or source code |

---

###  You're done!  
You can now install your private package securely just like any public Python package.

