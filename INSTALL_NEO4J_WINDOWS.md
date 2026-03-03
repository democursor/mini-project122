# Installing Neo4j on Windows (Without Docker)

Since Docker is not installed, use **Neo4j Desktop** - it's easier for Windows!

## Option 1: Neo4j Desktop (Recommended for Windows)

### Step 1: Download Neo4j Desktop
1. Go to: https://neo4j.com/download/
2. Click "Download Neo4j Desktop"
3. Fill in the form (or skip with fake email)
4. Download the installer (Neo4j Desktop Setup.exe)

### Step 2: Install Neo4j Desktop
1. Run the downloaded installer
2. Follow the installation wizard
3. Launch Neo4j Desktop after installation

### Step 3: Create a Database
1. Click "New" → "Create project"
2. Name it "Research Literature"
3. Click "Add" → "Local DBMS"
4. Set:
   - Name: `research-graph`
   - Password: `password`
   - Version: Latest (5.x)
5. Click "Create"

### Step 4: Start the Database
1. Click the "Start" button on your database
2. Wait for it to show "Active" (green dot)
3. Note the connection details:
   - Bolt URL: `bolt://localhost:7687`
   - Username: `neo4j`
   - Password: `password`

### Step 5: Verify Connection
```bash
python verify_phase3_setup.py
```

Should show: **5/5 checks passed** ✓

---

## Option 2: Neo4j Community Edition (Standalone)

### Step 1: Download
1. Go to: https://neo4j.com/deployment-center/
2. Select "Community Server"
3. Download Windows ZIP file

### Step 2: Extract
1. Extract ZIP to `C:\neo4j`
2. Open PowerShell as Administrator

### Step 3: Configure
Edit `C:\neo4j\conf\neo4j.conf`:
```
# Uncomment these lines:
dbms.default_listen_address=0.0.0.0
dbms.connector.bolt.listen_address=:7687
dbms.connector.http.listen_address=:7474
```

### Step 4: Set Initial Password
```powershell
cd C:\neo4j
.\bin\neo4j-admin.bat set-initial-password password
```

### Step 5: Start Neo4j
```powershell
.\bin\neo4j.bat console
```

Keep this window open while using Neo4j.

### Step 6: Verify
Open browser: http://localhost:7474
- Username: `neo4j`
- Password: `password`

---

## Option 3: Install Docker Desktop (For Future Use)

If you want to use Docker in the future:

### Step 1: Download Docker Desktop
https://www.docker.com/products/docker-desktop/

### Step 2: Install
1. Run installer
2. Restart computer when prompted
3. Launch Docker Desktop
4. Wait for it to start (whale icon in system tray)

### Step 3: Start Neo4j
```powershell
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
```

---

## Quick Verification

After installing Neo4j (any method), run:

```bash
# 1. Verify setup
python verify_phase3_setup.py

# 2. Build graph
python build_graph.py

# 3. Run tests
python test_phase3.py
```

---

## Troubleshooting

### "Connection refused" error
- Make sure Neo4j is started
- Check Neo4j Desktop shows "Active" status
- Verify port 7687 is not blocked by firewall

### "Authentication failed" error
- Password should be: `password`
- Or update `config/default.yaml` with your password

### Neo4j Desktop won't start
- Check system requirements (Windows 10+, 4GB RAM)
- Try running as Administrator
- Check antivirus isn't blocking it

---

## Recommended: Neo4j Desktop

For Windows users without Docker, **Neo4j Desktop is the easiest option**:
- ✓ Simple GUI interface
- ✓ No command line needed
- ✓ Easy to start/stop
- ✓ Built-in browser
- ✓ No configuration needed

**Download now:** https://neo4j.com/download/

---

## Next Steps

Once Neo4j is running:

1. **Verify:** `python verify_phase3_setup.py`
2. **Build graph:** `python build_graph.py`
3. **Explore:** Open http://localhost:7474
4. **Run queries:** Try the examples in PHASE3_QUICKSTART.md

---

**Need help?** Check PHASE3_SETUP_STATUS.md for detailed instructions.
