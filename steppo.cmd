@echo off
rem Windows wrapper: runs `steppo` (and a few helpers) inside the dev container,
rem starting it first if needed. Uses Podman when installed, otherwise Docker;
rem set STEPPO_ENGINE=docker (or podman) to choose. See docs/installation.md.
setlocal
pushd "%~dp0"

if defined STEPPO_ENGINE (
    set "ENGINE=%STEPPO_ENGINE%"
) else (
    where podman >nul 2>nul && (set "ENGINE=podman") || (set "ENGINE=docker")
)

set "COMPOSE=%ENGINE% compose -f docker\docker-compose.yml"
if /i "%ENGINE%"=="podman" (
    rem Podman ignores compose's `deploy` GPU reservation; the override adds the CDI device.
    set "COMPOSE=%COMPOSE% -f docker\docker-compose.podman.yml"
    set "PODMAN_COMPOSE_WARNING_LOGS=false"
)

if "%~1"=="" goto usage
if /i "%~1"=="help" goto usage
if /i "%~1"=="setup-gpu" goto setup_gpu

if /i "%ENGINE%"=="podman" (
    podman info >nul 2>nul || podman machine start || goto fail
)

if /i "%~1"=="stop" (
    %COMPOSE% down
    goto done
)

set "RUNNING="
for /f %%i in ('%COMPOSE% ps -q --status running app 2^>nul') do set "RUNNING=1"
if not defined RUNNING (
    echo Starting the StePPO container with %ENGINE%. The first start builds the image
    echo and downloads several GB; this can take 10 minutes or more.
    %COMPOSE% up -d || goto fail
)

set "ENVARGS="
if defined GPUS set "ENVARGS=-e GPUS=%GPUS%"

if /i "%~1"=="shell" (
    %COMPOSE% exec %ENVARGS% app bash
    goto done
)
if /i "%~1"=="jupyter" goto jupyter
%COMPOSE% exec %ENVARGS% app uv run steppo %*
goto done

:jupyter
rem A rootful Podman machine forwards ports with NAT rules, which WSL does not
rem relay to Windows localhost, so point the printed link at the VM's address.
set "HOST=127.0.0.1"
if /i "%ENGINE%"=="podman" (
    for /f "tokens=4 delims=/ " %%a in ('podman machine ssh "ip -4 -o addr show eth0"') do set "HOST=%%a"
)
echo Open the http://%HOST%:8888/lab?token=... link printed below. Stop with Ctrl+C.
%COMPOSE% exec %ENVARGS% app uv run --with jupyterlab jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root --ServerApp.custom_display_url=http://%HOST%:8888
goto done

:setup_gpu
if /i not "%ENGINE%"=="podman" (
    echo setup-gpu is only needed for Podman; Docker Desktop exposes the GPU itself.
    goto done
)
podman info >nul 2>nul || podman machine start || goto fail
podman machine ssh "curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null" || goto fail
podman machine ssh "sudo dnf install -y -q nvidia-container-toolkit" || goto fail
podman machine ssh "sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml" || goto fail
podman run --rm --device nvidia.com/gpu=all docker.io/library/ubuntu:22.04 nvidia-smi -L || goto fail
echo GPU setup done. Re-run `steppo setup-gpu` after updating the NVIDIA driver.
goto done

:usage
echo Usage: steppo ^<command^> [args]   (runs inside the dev container)
echo.
echo   solve ...    StePPO CLI, e.g. `steppo solve van_der_pol --xi 50 --plot traj.png`
echo                (paths are relative to the repository root; see `steppo solve -h`)
echo   jupyter      start JupyterLab on http://127.0.0.1:8888
echo   shell        open a bash shell in the container
echo   stop         stop the container
echo   setup-gpu    one-time GPU setup for Podman (and after NVIDIA driver updates)
echo.
echo Engine: %ENGINE% (set STEPPO_ENGINE=docker or podman to override)
goto done

:fail
popd
exit /b 1

:done
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
